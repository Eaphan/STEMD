from efg.data.datasets.waymo import WaymoDetectionDataset
from efg.data.registry import DATASETS
from efg.engine.registry import TRAINERS
from efg.data.registry import PROCESSORS
from efg.engine.trainer import DefaultTrainer
from efg.data.augmentations import AugmentationBase
from efg.data.voxel_generator import VoxelGenerator

import numpy as np
from copy import deepcopy
import os

@TRAINERS.register()
class CustomTrainer(DefaultTrainer):
    def __init__(self, configuration):
        super(CustomTrainer, self).__init__(configuration)
        if self.is_train:
            self.fade_start_iter = int(self.max_iters * (1 - self.config.trainer.fade))

    def step(self):
        if (
            self.iter > self.fade_start_iter
            and len(self.dataloader.dataset.transforms) == self.dataloader.dataset.transforms_length
        ):
            self.dataloader.dataset.transforms = self.dataloader.dataset.transforms[1:]
            self._dataiter = iter(self.dataloader)

        super().step()


@DATASETS.register()
class CustomWDDataset(WaymoDetectionDataset):
    def __init__(self, config):
        super(CustomWDDataset, self).__init__(config)
        self.transforms_length = len(self.transforms)


@PROCESSORS.register()
class CustomMultiFrameVoxelization(AugmentationBase):
    # params include nsweeps
    def __init__(self, pc_range, voxel_size, max_points_in_voxel, max_voxel_num, nsweeps):
        super().__init__()
        self._init(locals())
        self.voxel_generator = VoxelGenerator(
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            max_num_points=max_points_in_voxel,
            max_voxels=max_voxel_num,
        )
        self.nsweeps = nsweeps

    def __call__(self, points, info):
        # [0, -40, -3, 70.4, 40, 1]
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size
        voxels_list = [] # list item by time
        coordinates_list = []
        num_points_per_voxel_list = []

        # keyframe_num = sum(['annotations' in x for x in info['sweeps']])
        # unique_frames = []
        # seen_paths = set()
        # for frame in x:
        #     if frame['lidar_path'] not in seen_paths:
        #         unique_frames.append(frame)
        #         seen_paths.add(frame['lidar_path'])
        #     else:
        #         break
        # unique_frame_num = len(unique_frames)

        seq_len = 4
        keyframe_interval = self.nsweeps // seq_len

        keyframe_annotations = []
        is_training = 'annotations' in info
        if is_training: # True when training
            keyframe_annotations.append(info['annotations'])
        time_lag_list = []

        last_lidar_path = info["path"]
        for i in range(self.nsweeps-1):
            sweep = info["sweeps"][i]
            if sweep["path"]==last_lidar_path:
                break
            last_lidar_path = sweep["path"]

            if (i+1)%keyframe_interval==0:
                assert 'annotations' in sweep
                keyframe_annotations.append(sweep['annotations'])
                time_lag_list.append(sweep['time_lag'])
            
                if len(keyframe_annotations)==seq_len:
                    break

        if is_training:
            while len(keyframe_annotations) < seq_len:
                keyframe_annotations.append(keyframe_annotations[-1])

            info['keyframe_annotations'] = keyframe_annotations

        # for current frame
        time_interval = 0.1 * keyframe_interval - 0.001
        current_points = points[(points[:, -1] >=0)&(points[:, -1] <= 0+time_interval), :]
        voxels, coordinates, num_points_per_voxel = self.voxel_generator.generate(current_points)
        voxels_list.append(voxels)
        coordinates_list.append(coordinates)

        num_points_per_voxel_list.append(num_points_per_voxel)

        # import pdb;pdb.set_trace()
        for i in range(seq_len-1):
            if i < len(time_lag_list):
                keyframe_time_lag = time_lag_list[i]
                current_points = deepcopy(points[(points[:, -1] >= keyframe_time_lag)&(points[:, -1] <= keyframe_time_lag+time_interval)])
                current_points[:, -1] = current_points[:, -1] - keyframe_time_lag
                voxels, coordinates, num_points_per_voxel = self.voxel_generator.generate(current_points)
                voxels_list.append(voxels)
                coordinates_list.append(coordinates)
                num_points_per_voxel_list.append(num_points_per_voxel)
            else:
                voxels_list.append(voxels)
                coordinates_list.append(coordinates)
                num_points_per_voxel_list.append(num_points_per_voxel)


        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        point_voxels = dict(
            voxels_list=voxels_list,
            points=points,
            coordinates_list=coordinates_list,
            num_points_per_voxel_list=num_points_per_voxel_list,
            num_voxels=num_voxels,
            shape=grid_size,
            range=pc_range,
            size=self.voxel_size,
        )
        return point_voxels, info

