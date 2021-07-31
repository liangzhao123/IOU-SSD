from easydict import EasyDict as edict
import os

import pickle
import vis_points.utils as utils

import numpy as np
from pvdet.dataset.utils.box_utils import boxes3d_to_corners3d_lidar
import cv2
from vis_points.vis_fps_with_features import get_labels
from vis_points.vis_config import cfg_vis
from vis_points.vis_fps_with_features import fps_d
import mayavi.mlab as mlab


def show_predicted_results(sample_id_in_val=None,draw_text=True):

    if show_predicted_results is None:
        sample_id_in_val = cfg_vis.val_idx
    pred_path = cfg_vis.predict_path
    val_split_dir = cfg_vis.val_dataset
    val_sample_idx = [x.strip() for x in open(val_split_dir).readlines()]
    gt_lidar_dir = os.path.join(cfg_vis.kitti_gt_dir, "lidar_gt_list.pth")
    with open(gt_lidar_dir, "rb") as f:
        gt_lidar_list_with_class = pickle.load(f)
    gt_lidar_list = gt_lidar_list_with_class["gt_lidar_box_list"]
    gt_class_names_list = gt_lidar_list_with_class["gt_class_names"]
    sample_idx_str = val_sample_idx[sample_id_in_val]
    sample_idx = int(sample_idx_str)

    points_path = os.path.join(cfg_vis.KITTI_DIR, "velodyne", sample_idx_str + ".bin")
    point_xyzi = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)

    image_path = os.path.join(cfg_vis.KITTI_DIR, "image_2", sample_idx_str + ".png")
    image = cv2.imread(image_path)

    with open(pred_path, "rb") as f:
        pred_dicts = pickle.load(f)
    det_anno = pred_dicts[sample_id_in_val]

    det_boxes = det_anno['boxes_lidar']
    det_class_name = list(det_anno["name"])

    gt_boxes = gt_lidar_list[sample_id_in_val]
    gt_classes = gt_class_names_list[sample_id_in_val]

    gt_boxes_corners = boxes3d_to_corners3d_lidar(gt_boxes[:, :7])
    det_boxes_corners = boxes3d_to_corners3d_lidar(det_boxes[:, :7])

    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
    )

    fig = utils.draw_gt_boxes3d(gt_boxes_corners, color=(1, 0, 0), fig=fig)

    if draw_text:
        fig = utils.draw_gt_boxes3d(det_boxes_corners, color=(0, 1, 0), label=det_class_name, draw_text=True, fig=fig)
    else:
        fig = utils.draw_gt_boxes3d(det_boxes_corners, color=(0, 1, 0), fig=fig)

    fig = utils.draw_lidar(point_xyzi[:, 0:3], fig=fig)
    cv2.imshow("image", image)
    mlab.show()


if __name__ == '__main__':
    pass
    # single_stage_mode()