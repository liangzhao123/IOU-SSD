import numpy as np
import torch
from single_stage_model.dataset.leishen_dataset.leishen_ops import leishen_opts_utils
from scipy.spatial import Delaunay
import scipy
def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object(line) for line in lines]
    return objects

def get_info(objects):
    annotations = {}
    gt_boxes_list = []
    for object in objects:
        gt_box = [*object.loc,object.w,object.l,object.h,object.ry,object.cls_id]
        gt_boxes_list.append(gt_box)
    return np.array(gt_boxes_list)


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Rider': 3, 'Bus': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

class Object(object):
    def __init__(self,line):
        super().__init__()
        label = line.strip().split(' ')
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.w = float(label[4])
        self.l = float(label[5])
        self.h = float(label[6])
        self.ry = - float(label[7])
        self.loc = np.array((float(label[1]), float(label[2]), float(label[3])-self.h/2.0), dtype=np.float32)

def read_bin(path):
    points  = np.fromfile(path,dtype=np.float32).reshape(-1,4)
    return points



def center2corner_leishen(boxes3d, bottom_center=True,return_temp_corner=True):
    """
        :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords, see the definition of ry in KITTI dataset
        :param z_bottom: whether z is on the bottom center of object
        :return: corners3d: (N, 8, 3)
            7 -------- 4
           /|         /|
          6 -------- 5 .
          | |        | |
          . 3 -------- 0
          |/         |/
          2 -------- 1
        """
    boxes_num = boxes3d.shape[0]

    w, l, h = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    y_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    x_corners= np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.], dtype=np.float32).T
    if bottom_center:
        z_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        z_corners[:, 4:8] = h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        z_corners = np.array([-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), -np.sin(ry), zeros],
                         [np.sin(ry), np.cos(ry), zeros],
                         [zeros, zeros, ones]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)


    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    rotated = False
    if rotated:
        x_corners, y_corners, z_corners = temp_corners[:, :, 0], temp_corners[:, :, 1], temp_corners[:, :, 2]
    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)
    if return_temp_corner:
        return corners.astype(np.float32),temp_corners.astype(np.float32)
    else:
        return corners.astype(np.float32)

def remove_points_in_boxes3d_v0(points, boxes3d):
    """
    :param points: (npoints, 3 + C)
    :param boxes3d: (N, 7) [x, y, z, w, l, h, rz] in LiDAR coordinate, z is the bottom center, each box DO NOT overlaps
    :return:
    """
    point_masks = leishen_opts_utils.points_in_boxes_cpu(
        torch.from_numpy(points[:, 0:3]), torch.from_numpy(boxes3d)
    ).numpy()
    return points[point_masks.sum(axis=0) == 0]

def mask_boxes_outside_range(boxes, limit_range):
    """
    :param boxes: (N, 7) (N, 7) [x, y, z, w, l, h, r] in LiDAR coords
    :param limit_range: [minx, miny, minz, maxx, maxy, maxz]
    :return:
    """
    corners3d,_ = center2corner_leishen(boxes)  # (N, 8, 3)
    mask = ((corners3d >= limit_range[0:3]) & (corners3d <= limit_range[3:6])).all(axis=2)

    return mask.sum(axis=1) == 8  # (N)

def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag