import torch
import torch.nn as nn
from torch.autograd import Function
import leishen_ops_cuda



def points_in_boxes_gpu(points, boxes):
    """
    :param points: (B, M, 3)
    :param boxes: (B, T, 8), num_valid_boxes <= T
    :return box_idxs_of_pts: (B, M), default background = -1
    """
    assert boxes.shape[0] == points.shape[0]
    assert boxes.shape[2] == 7
    batch_size, num_points, _ = points.shape

    box_idxs_of_pts = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
    leishen_ops_cuda.points_in_boxes_gpu(boxes.contiguous(), points.contiguous(), box_idxs_of_pts)

    return box_idxs_of_pts


def points_in_boxes_cpu(points, boxes):
    """
    :param points: (npoints, 3)
    :param boxes: (N, 7) [x, y, z, w, l, h, rz] in LiDAR coordinate, z is the bottom center, each box DO NOT overlaps
    :return point_indices: (N, npoints)
    """
    assert boxes.shape[1] == 7
    assert points.shape[1] == 3

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
    leishen_ops_cuda.points_in_boxes_cpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

    return point_indices


if __name__ == '__main__':
    pass
