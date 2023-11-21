import copy

import torch
from torch import nn
import numpy as np

from efg.data.datasets.waymo import collate
from efg.modeling.backbones.fpn import build_resnet_fpn_backbone
from efg.modeling.readers.voxel_reader import VoxelMeanFeatureExtractor
from efg.modeling.operators.iou3d_nms import boxes_iou_bev

from cdn import dn_post_process, prepare_for_cdn
from heads import Det3DHead
from modules.backbone3d import Backbone3d
from modules.box_coder import VoxelBoxCoder3D
from transformer import Transformer
from modules.temporal_conv_gru import TemporalConvGRU
from modules.utils import get_clones

from torch_geometric.nn import GCNConv, GATv2Conv
import torch.nn.functional as F
from collections import OrderedDict

class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in, dim_h, dim_out, heads=8, add_self_loops=True):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_out//heads, heads=heads, add_self_loops=add_self_loops)
        # self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1, add_self_loops=add_self_loops)
        #self.optimizer = torch.optim.Adam(self.parameters(),
        #                                  lr=0.005,
        #                                  weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = h + x
        #h = F.dropout(h, p=0.6, training=self.training)
        #h = self.gat2(h, edge_index)
        #h = h + x
        return h

class STEMD(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = torch.device(config.model.device)

        # setup properties
        self.seq_len = 4
        self.hidden_dim = config.model.hidden_dim
        self.aux_loss = config.model.aux_loss
        self.num_classes = len(config.dataset.classes)
        self.num_queries = config.model.transformer.num_queries

        # build backbone
        # input_dim = len(config.dataset.format) if config.dataset. nsweeps == 1 else len(config.dataset.format) + 1
        input_dim = len(config.dataset.format)
        reader = VoxelMeanFeatureExtractor(**config.model.backbone.reader, num_input_features=input_dim)
        extractor = build_resnet_fpn_backbone(config.model.backbone.extractor, input_dim)
        self.backbone = Backbone3d(
            config.model.backbone.hidden_dim,
            reader,
            extractor,
            config.model.backbone.position_encoding,
            out_features=config.model.backbone.out_features,
        )
        in_channels = self.backbone.num_channels

        # build input projection from backbone to transformer
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[i], self.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, self.hidden_dim),
            )
            for i in range(len(self.backbone.out_features))
        ])
        for module in self.input_proj.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # build transformer
        self.transformer = Transformer(
            d_model=config.model.transformer.hidden_dim,
            nhead=config.model.transformer.nhead,
            nlevel=len(config.model.backbone.out_features),
            num_encoder_layers=config.model.transformer.enc_layers,
            num_decoder_layers=config.model.transformer.dec_layers,
            dim_feedforward=config.model.transformer.dim_feedforward,
            dropout=config.model.transformer.dropout,
            num_queries=config.model.transformer.num_queries,
            num_classes=self.num_classes,
            mom=config.model.contrastive.mom,
        )
        self.transformer.proposal_head = Det3DHead(
            config,
            with_aux=False,
            with_metrics=False,
            num_classes=1,
            num_layers=1,
        )
        self.transformer.decoder.detection_head = Det3DHead(
            config,
            with_aux=True,
            with_metrics=False,
            num_classes=len(config.dataset.classes),
            num_layers=config.model.transformer.dec_layers,
        )
        # self.transformer.decoder_gt = copy.deepcopy(self.transformer.decoder)
        # for param_q, param_k in zip(self.transformer.decoder.parameters(), self.transformer.decoder_gt.parameters()):
        #     param_k.data.copy_(param_q.data)  # initialize
        #     param_k.requires_grad = False  # not update by gradient

        self.transformer.graph_head = Det3DHead(
            config,
            with_aux=False,
            with_metrics=True,
            num_classes=len(config.dataset.classes),
            num_layers=1,
        )
        self.reused_graph_query_num = 300

        self.st_block_num = 1
        st_block = nn.Sequential(OrderedDict([
            ('spatial_transformer', GAT(256, 256//8, 256)),
            ('temporal_transformer', GAT(256, 256//8, 256, add_self_loops=False)),
            # ('spatial_transformer_2', GAT(256, 256//8, 256)),
            # ('temporal_transformer_2', GAT(256, 256//8, 256, add_self_loops=False))
            ]))
        self.st_blocks = get_clones(st_block, self.st_block_num)

        # to initialize the GRU and temporal deformable attention
        channels = config.model.transformer.hidden_dim
        hidden_dim = [config.model.transformer.hidden_dim, config.model.transformer.hidden_dim]
        self.temporal_conv_gru = TemporalConvGRU(
            input_size=(188, 188), # ad hoc
            input_dim=channels,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3), # kernel size for two stacked hidden layers
            num_layers=2, # ad hoc
            dtype=torch.cuda.FloatTensor,
            batch_first=False, # ?
            bias = True,
            return_all_layers = False,
            deform_encoder_config = config.model.temporal_conv_gru.deform_encoder
        )


        # build annotaion coder
        self.box_coder = VoxelBoxCoder3D(
            config.dataset.voxel_size,
            config.dataset.pc_range,
            device=self.device,
        )

        # contrastive projector
        contras_dim = config.model.contrastive.dim
        self.eqco = config.model.contrastive.eqco
        self.tau = config.model.contrastive.tau
        self.contras_loss_coeff = config.model.contrastive.loss_coeff
        self.projector = nn.Sequential(
            nn.Linear(10, contras_dim),
            nn.ReLU(),
            nn.Linear(contras_dim, contras_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(contras_dim, contras_dim),
            nn.ReLU(),
            nn.Linear(contras_dim, contras_dim),
        )
        self.similarity_f = nn.CosineSimilarity(dim=2)

        self.config = config
        self.to(self.device)

    def forward(self, batched_inputs):

        batch_size = len(batched_inputs)

        # samples: ['voxels', 'points', 'coordinates', 'num_points_per_voxel', 'num_voxels', 'shape', 'range', 'size']
        samples = collate([bi[0] for bi in batched_inputs], self.device)

        if self.training:
            targets_list = []
            for seq_idx in range(self.seq_len):
                targets = [bi[1]["keyframe_annotations"][seq_idx] for bi in batched_inputs]
                for i in range(batch_size):
                    for key in ['gt_boxes', 'difficulty', 'num_points_in_gt', 'labels']:
                        # key = f'{ori_key}_{self.seq_len-2-t}'
                        targets[i][key] = torch.tensor(targets[i][key], device=self.device)

                targets = [self.box_coder.encode(tgt) for tgt in targets]
                targets_list.append(targets)
            targets_list=targets_list[::-1]
        else:
            targets_list = None

        # import pdb;pdb.set_trace()
        features_list = []
        pos_encodings_list = []
        memory_list = []
        for t in range(self.seq_len-1, -1, -1):
            voxels, coords, num_points_per_voxel, input_shape = (
                samples["voxels_list"][t],
                samples["coordinates_list"][t],
                samples["num_points_per_voxel_list"][t],
                samples["shape"][0],
            )

            ms_backbone_features_with_pos_embed = self.backbone(
                voxels, coords, num_points_per_voxel, batch_size, input_shape
            )

            features = []
            pos_encodings = []
            for idx, feat_pos in enumerate(ms_backbone_features_with_pos_embed):
                features.append(self.input_proj[idx](feat_pos[0]))
                pos_encodings.append(feat_pos[1])
            features_list.append(features)
            pos_encodings_list.append(pos_encodings)

        # import pdb;pdb.set_trace()
        # outputs = self.transformer(features, pos_encodings)
        src_anchors = self.transformer._create_ref_windows(features_list[0])

        # len of features_list=4, featurse_list[0][0] 
        bs, c, fe_h, fe_w = features_list[0][0].shape

        for t in range(self.seq_len):
            src, pos = features_list[t], pos_encodings_list[t]
            # here we implement the encoder and decoder of transformer
            assert pos is not None, "position encoding is required!"
            from modules.utils import flatten_with_shape
            src, _, src_shape = flatten_with_shape(src, None)
            src_pos = []
            for pe in pos:
                b, c = pe.shape[:2]
                pe = pe.view(b, c, -1).transpose(1, 2)
                src_pos.append(pe)
            src_pos = torch.cat(src_pos, dim=1)
            src_start_index = torch.cat([src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]])

            # import pdb;pdb.set_trace()
            memory = self.transformer.encoder(src, src_pos, src_shape, src_start_index, src_anchors)
            memory_list.append(memory.view(batch_size, fe_h, fe_w, -1)) # [2, 35344, 256] -。 [2, 188, 188, 256]
        
        # todo, temporal deformable attention and GRU to get new memory
        gru_input = torch.stack(memory_list, 0).permute(0, 1, 4, 2, 3) # (4, 2, 188, 188, 256) -> (4, 2, 256, 188, 188)
        new_stack_memory, last_state = self.temporal_conv_gru(gru_input) # new_stack_memory [batch_size, seq_len, output_dim, h, w]

        hs_list = []
        select_boxes_list = []
        topk_indexes_list = []
        inter_references_list = []
        inter_references_logits_list = []
        inter_references_gt_list = []
        inter_references_logits_gt_list = []
        dn_meta_list = []
        st_block_output = [] # 每个元素中是单帧的每个block的输出
        graph_out_class_list = []
        graph_out_coords_list = []
        select_box_num = 800

        dn = self.config.model.dn
        for t in range(self.seq_len):
            # prepare_for_cdn
            if self.training and dn.enabled and dn.dn_number > 0:
                # input_query_label, input_query_bbox, attn_mask, dn_meta = prepare_for_cdn(
                noised_gt_onehot, noised_gt_box, attn_mask, dn_meta = prepare_for_cdn(
                    dn_args=(targets_list[t], dn.dn_number, dn.dn_label_noise_ratio, dn.dn_box_noise_scale),
                    training=self.training,
                    num_queries=self.num_queries+self.reused_graph_query_num, #？
                    num_classes=self.num_classes,
                    hidden_dim=self.hidden_dim,
                    label_enc=None,
                )
            else:
                # input_query_bbox = input_query_label = attn_mask = dn_meta = None
                noised_gt_onehot = noised_gt_box = attn_mask = dn_meta = None


            t_memory = new_stack_memory[:, t].permute(0, 2, 3, 1).view(bs, -1, c) # [bs, 35344, c]

            # None, None, [bs, num_queries, 10] (后三维都是一样的，表示前景的概率), [bs, num_queries, 1]
            if t == 0:
                query_embed, query_pos, topk_proposals, topk_indexes = self.transformer._get_enc_proposals(t_memory, src_anchors, spec_num_queries=self.num_queries + self.reused_graph_query_num)
            else:
                query_embed, query_pos, topk_proposals, topk_indexes = self.transformer._get_enc_proposals(t_memory, src_anchors)

            # if t!=0:
            #     decoder_input = torch.cat([topk_proposals, topk_graph_output_topk], 1)
            # else:
            #     decoder_input = topk_proposals            

            # # hs: [decoder_layer_num, bs, num_queries, c], [decoder_layer_num, bs, num_queries, 7], [decoder_layer_num, bs, num_queries, 3]
            # hs, inter_references, inter_references_logits = self.transformer.decoder(query_embed, query_pos, t_memory, src_shape, src_start_index, decoder_input)

            if self.training and noised_gt_box is not None and noised_gt_box.shape[1]!=0:
                noised_gt_proposals = torch.cat(
                    (
                        noised_gt_box,
                        noised_gt_onehot,
                    ),
                    dim=-1,
                )
                topk_proposals = torch.cat(
                    (
                        noised_gt_proposals,
                        topk_proposals,
                    ),
                    dim=1,
                )
            # init_reference_out = topk_proposals[..., :7]

            if t!=0:
                decoder_input = torch.cat([topk_proposals, topk_graph_output_topk], 1)
            else:
                decoder_input = topk_proposals     

            hs, inter_references, inter_references_logits = self.transformer.decoder(query_embed, query_pos, t_memory, src_shape, 
                                src_start_index, decoder_input, attn_mask)
            if self.training:
                hs = hs[:, :, dn_meta["pad_size"]:, :]

            if self.training and dn.dn_number > 0 and dn_meta is not None:
                inter_references_logits, inter_references = dn_post_process(
                    inter_references_logits,
                    inter_references,
                    dn_meta,
                    self.aux_loss,
                    self._set_aux_loss,
                )
                dn_meta_list.append(dn_meta)
            # todo: process the out of dn_metas

            ##################################### todo: graph pooling here ########################################################
            hs_single_hidden_select_bs_list = [] # item, (new_n, C)
            select_references_bs_list = [] # item (new_n, 7)
            select_boxes_bs_list = [] # item (new_n, 7)

            for i in range(batch_size):
                cur_boxes = inter_references[-1][i].clone() # (N, 7)
                cur_conf = inter_references_logits[-1][i].max(1)[0].sigmoid()
                cur_num_queries = cur_boxes.shape[0]
                cur_boxes[:, [0, 3]] *= self.config.dataset.pc_range[3] - self.config.dataset.pc_range[0]
                cur_boxes[:, [1, 4]] *= self.config.dataset.pc_range[4] - self.config.dataset.pc_range[1]
                cur_boxes[:, [2, 5]] *=self.config.dataset.pc_range[5] - self.config.dataset.pc_range[2]
                cur_boxes[:, 6] *= np.pi * 2
                bev_ious = boxes_iou_bev(cur_boxes, cur_boxes)
                bev_ious[torch.arange(cur_num_queries), torch.arange(cur_num_queries)] = 0
                conf_repeat = cur_conf.view(1, -1).repeat(cur_num_queries, 1)
                neighbor_max_conf, neighbor_max_conf_index = ((bev_ious > 0.5) * conf_repeat).max(1)
                neighbor_max_conf_index_iou = bev_ious.gather(1, neighbor_max_conf_index.view(-1, 1)).flatten()
                # cur_conf = cur_conf * (1 - (cur_conf < neighbor_max_conf) * neighbor_max_conf_index_iou)
                cur_conf = cur_conf * (1 - (cur_conf < neighbor_max_conf) * neighbor_max_conf_index_iou)
                # print("### cur_conf == some, count", (cur_conf==some).sum())

                _, indexes = torch.topk(cur_conf, select_box_num, sorted=False)
                hs_single_hidden_select = hs[-1][i][indexes, :]
                cur_boxes_select = cur_boxes[indexes, :]
                references_select = inter_references[-1][i][indexes, :]

                select_boxes_bs_list.append(cur_boxes_select)
                select_references_bs_list.append(references_select)
                hs_single_hidden_select_bs_list.append(hs_single_hidden_select)

            select_boxes = torch.stack(select_boxes_bs_list, dim=0) # (bs,n,7)
            select_references = torch.stack(select_references_bs_list, dim=0)
            select_hs = torch.stack(hs_single_hidden_select_bs_list, dim=0)

            select_boxes_list.append(select_boxes)
            hs_list.append(select_hs)
            topk_indexes_list.append(topk_indexes)
            inter_references_list.append(inter_references)
            inter_references_logits_list.append(inter_references_logits)

            last_select_boxes = select_boxes if t==0 else select_boxes_list[-2]

            # to know the adjacent matrix of proposals, combine multiple graphs into one large graph.
            # add_self_loops set False
            # self.spatial_temporal_block()
            dist_threshold = 2
            cur_refer_box_center = select_boxes[..., :2] # (bs, num_queries, 2)
            dist = torch.cdist(cur_refer_box_center, cur_refer_box_center) # (bs, num_queries, num_queries)
            def merge_subgraph_into_graph(sub_graphs):
                # sub_graphs [bs, n, n]
                bs, n, _ = sub_graphs.shape
                out_dim = bs * n
                # output_matrix = torch.new_zeros([out_dim, out_dim])
                edge_list = []
                for i in range(bs):
                    adjacent_m = torch.nonzero(sub_graphs[i]<dist_threshold)
                    adjacent_m += i * n
                    edge_list.append(adjacent_m)
                big_graph_edges = torch.cat(edge_list)
                return big_graph_edges

            spatial_edge_index = merge_subgraph_into_graph(dist)

            # another cross-attention (temporal)
            ## first compute the adjacent matrix
            temporal_edge_list = []
            temporal_dist_threshold = 2
            last_refer_box_center = last_select_boxes[..., :2].clone() # (bs, num_queries, 2)

            for i in range(bs):
                single_last_refer_box_center = last_refer_box_center[i]
                single_cur_refer_box_center = cur_refer_box_center[i]
                try:
                    concat_centers = torch.cat([single_last_refer_box_center, single_cur_refer_box_center])
                except:
                    import pdb;pdb.set_trace()
                dist = torch.cdist(concat_centers, concat_centers) # return (2N, 2N)

                dist[:, :select_box_num] = 99
                dist[select_box_num:, :] = 99
                dist[torch.arange(select_box_num, select_box_num*2), torch.arange(select_box_num, select_box_num*2)] = 0
                single_adjacent_m = torch.nonzero(dist < temporal_dist_threshold)
                single_adjacent_m[single_adjacent_m>=select_box_num] += (bs-1) * select_box_num
                single_adjacent_m += i * select_box_num
                # single_adjacent_m[:, 1] += (bs-1) * select_box_num
                temporal_edge_list.append(single_adjacent_m)
            temporal_graph_edges = torch.cat(temporal_edge_list)
            
            frame_block_outputs = []
            for st_index in range(self.st_block_num):
                if t==0:
                    if st_index==0:
                        last_hs = select_hs # decoder output
                    else:
                        last_hs = frame_block_outputs[-1] # 当前帧 上一个block的输出
                else:
                    if st_index==0:
                        last_hs = hs_list[-2]#上一帧 decoder的输出
                    else:
                        last_hs = st_block_output[t-1][st_index-1] #上一帧 上block的输出

                if st_index==0:
                    node_feas = select_hs.view(-1, hs.shape[-1]) # decoder output
                else:
                    node_feas = frame_block_outputs[-1].view(-1, hs.shape[-1]) # 当前帧上一个block的输出

                new_spatial_feas = self.st_blocks[st_index].spatial_transformer(node_feas, spatial_edge_index.T) # return (bs*nquery, c)

                temporal_graph_node_feas = torch.cat(
                    [last_hs.view(-1, last_hs.shape[-1]),
                    node_feas.view(-1, hs.shape[-1])],
                )
                new_temporal_feas = self.st_blocks[st_index].temporal_transformer(temporal_graph_node_feas, temporal_graph_edges.T) # return (bs*nquery*2, c)
                new_temporal_feas_valid = new_temporal_feas[bs * select_box_num:, :] # ad hoc
                
                spatial_feas_input_2 = new_spatial_feas + new_temporal_feas_valid
                # # spatial_feas_input_2 = torch.cat([new_spatial_feas, new_temporal_feas_valid], 1)
                # new_spatial_feas_2 = self.st_blocks[st_index].spatial_transformer_2(spatial_feas_input_2, spatial_edge_index.T) # return (bs*nquery, c)
                # temporal_graph_node_feas_2 = torch.cat(
                #     [last_hs.view(-1, last_hs.shape[-1]),
                #     new_spatial_feas_2],
                # )
                # new_temporal_feas_2 = self.st_blocks[st_index].temporal_transformer_2(temporal_graph_node_feas_2, temporal_graph_edges.T) # return (bs*nquery*2, c)
                # new_temporal_feas_valid_2 = new_temporal_feas_2[bs * select_box_num:, :] # ad hoc
                # block_output_embed = new_temporal_feas_valid_2.view(bs, select_box_num, c)

                block_output_embed = spatial_feas_input_2.view(bs, select_box_num, c)
                frame_block_outputs.append(block_output_embed)

            st_block_output.append(frame_block_outputs)
            graph_out_class, graph_out_coords = self.transformer.graph_head(frame_block_outputs[-1], select_references)
            graph_out_class_list.append(graph_out_class)
            graph_out_coords_list.append(graph_out_coords)

            _, indexes = torch.topk(graph_out_class.max(2)[0], self.reused_graph_query_num, dim=1, sorted=False)
            indexes = indexes.unsqueeze(-1) # (bs, 300, 1)
            topk_graph_output_box = torch.gather(graph_out_coords, 1, indexes.expand(-1, -1, graph_out_coords.shape[-1]))
            topk_graph_output_logits = torch.gather(graph_out_class, 1, indexes.expand(-1, -1, graph_out_class.shape[-1]))
            topk_graph_output_probs = topk_graph_output_logits.sigmoid()
            topk_graph_output_topk = torch.cat([topk_graph_output_box, topk_graph_output_probs], 2).detach()


        ###################### split line ###############################
        new_memory = new_stack_memory[:, -1].permute(0, 2, 3, 1).view(bs, -1, c) # [bs, 35344, c]

        # # None, None, [bs, num_queries, 10] (后三维都是一样的，表示前景的概率), [bs, num_queries, 1]
        # query_embed, query_pos, topk_proposals, topk_indexes = self.transformer._get_enc_proposals(new_memory, src_anchors)
        # init_reference_out = topk_proposals[..., :7]

        # # hs: [decoder_layer_num, bs, num_queries, c], [decoder_layer_num, bs, num_queries, 10]
        # hs, inter_references, inter_references_logits = self.transformer.decoder(query_embed, query_pos, new_memory, src_shape, src_start_index, topk_proposals)
        # inter_references_out = inter_references

        # 方便后面操作
        # return hs, init_reference_out, inter_references_out, memory, src_anchors, topk_indexes
        # src_embed, src_ref_windows, src_indexes = new_memory, src_anchors, topk_indexes

        # outputs_class = inter_references_logits
        # outputs_coord = inter_references

        # todo: dn post process
        # if dn.dn_number > 0 and dn_meta is not None:
        #     outputs_class, outputs_coord = dn_post_process(
        #         outputs_class,
        #         outputs_coord,
        #         dn_meta,
        #         self.aux_loss,
        #         self._set_aux_loss,
        #     )

        if self.training:
            losses = {}

            src_ref_windows = src_anchors
            for t in range(self.seq_len):
                targets = targets_list[t]
                dn_meta = dn_meta_list[t]

                # encoder loss
                src_indexes = topk_indexes_list[t]
                src_embed = new_stack_memory[:, t].permute(0, 2, 3, 1).view(bs, -1, c) # [bs, 35344, c]
                enc_class, enc_coords = self.transformer.proposal_head(src_embed, src_ref_windows)

                bin_targets = copy.deepcopy(targets)
                [tgt["labels"].fill_(0) for tgt in bin_targets]
                enc_outputs = {
                    "topk_indexes": src_indexes,
                    "pred_logits": enc_class,
                    "pred_boxes": enc_coords,
                }
                enc_losses = self.transformer.proposal_head.compute_losses(enc_outputs, bin_targets)
                losses.update({f"{k}_enc_t{t}": v for k, v in enc_losses.items()})

                # decoder loss
                # compute decoder losses
                outputs = {
                    "pred_logits": inter_references_logits_list[t][-1],
                    "pred_boxes": inter_references_list[t][-1],
                    "aux_outputs": self._set_aux_loss(inter_references_logits_list[t][:-1], inter_references_list[t][:-1]),
                }
                dec_losses = self.transformer.decoder.detection_head.compute_losses(outputs, targets, dn_meta=dn_meta, iou_reg=True)
                losses.update({f"{k}_dec_t{t}": v for k, v in dec_losses.items()})
            
                # compute contrastive loss
                # removed

                # compute loss for graph output
                graph_head_outputs = {
                    "pred_logits": graph_out_class_list[t],
                    "pred_boxes": graph_out_coords_list[t],
                }
                graph_head_losses = self.transformer.graph_head.compute_losses(graph_head_outputs, targets, iou_reg=True)
                losses.update({f"{k}_t{t}": v for k, v in graph_head_losses.items()})
            return losses
        else:
            # out_logits = outputs_class.squeeze()
            # out_bbox = outputs_coord.squeeze()

            # out_logits = outputs_class[-1]
            # out_bbox = outputs_coord[-1]

            out_logits = graph_out_class
            out_bbox = graph_out_coords

            out_prob = out_logits.sigmoid()
            out_prob = out_prob.view(out_logits.shape[0], -1)
            out_bbox = self.box_coder.decode(out_bbox)

            def _process_output(indices, bboxes):
                topk_boxes = indices.div(out_logits.shape[2], rounding_mode="floor")
                labels = indices % out_logits.shape[2]
                boxes = torch.gather(
                    bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, out_bbox.shape[-1])
                )
                return labels + 1, boxes, topk_boxes

            # scores, topk_indices = torch.topk(out_prob, 300, dim=1, sorted=False)
            # # VEHICLE: 0.15, PEDESTRIAN: 0.12, CYCLIST: 0.1
            topk_indices = torch.nonzero(out_prob >= 0.1, as_tuple=True)[1]
            scores = out_prob[:, topk_indices]

            labels, boxes, topk_indices = _process_output(topk_indices.view(1, -1), out_bbox)

            results = [
                {
                    "scores": s.detach().cpu(),
                    "labels": l.detach().cpu(),
                    "boxes3d": b.detach().cpu(),
                }
                for s, l, b in zip(scores, labels, boxes)
            ]
            return results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)
        ]

