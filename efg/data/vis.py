import copy

import numpy as np

import pylab as plt


def draw_box(ax, vertices, color="black"):
    """
    Draw box with color.
    Args:
        ax (list): axes to draw box along
        vertices (ndarray): indices of shape (N x 2)
        color (str): plotted color
    """
    connections = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ]
    for connection in connections:
        ax.plot(*vertices.transpose()[:, connection], c=color, lw=2)


def visualize_feature_maps(fm, boxes=[], boxes2=[], keypoints=[], stride=1, save_filename=None):
    """
    Visualize feature map with boxes or key points.
    Args:
        fm (torch.Tensor): feature map of shape H x W x c, c is channel
        boxes (ndarray): boxes to be visualized.
        keypoints (ndarray): key points to be visualized
        stride (int): used to normalize boxes or keypoints
        save_filename (bool): whether save to disk
    """
    nc = np.ceil(np.sqrt(fm.shape[2]))  # column
    nr = np.ceil(fm.shape[2] / nc)  # row
    nc = int(nc)
    nr = int(nr)
    plt.figure(figsize=(64, 64))
    for i in range(fm.shape[2]):
        ax = plt.subplot(nr, nc, i + 1)
        ax.imshow(fm[:, :, i], cmap="twilight")

        for obj in boxes:
            box = copy.deepcopy(obj) / stride
            draw_box(ax, box, color="g")

        for obj in boxes2:
            box = copy.deepcopy(obj) / stride
            draw_box(ax, box, color="y")

        for pts_score in keypoints:
            pts = pts_score[:8]
            pts = pts / stride
            for i in range(4):
                ax.plot(pts[2 * i + 1], pts[2 * i + 0], "r*")
            ax.plot([pts[1], pts[3]], [pts[0], pts[2]], c="y", lw=2)
            ax.plot([pts[3], pts[5]], [pts[2], pts[4]], c="g", lw=2)
            ax.plot([pts[5], pts[7]], [pts[4], pts[6]], c="b", lw=2)
            ax.plot([pts[7], pts[1]], [pts[6], pts[0]], c="r", lw=2)

        # plt.colorbar()
        ax.axis("off")
    if save_filename:
        plt.savefig(save_filename)
    else:
        plt.show()
    plt.close()
