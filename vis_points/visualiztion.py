
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
from vis_points.single_stage_model_utils.utils import show_predicted_results
import sys
import time
from vis_points.utils import kitti_object

def show_img_with_box(id= 211,show_raw=False,show_proposals=False):
    val_split_dir = cfg_vis.val_dataset
    val_sample_idx = [x.strip() for x in open(val_split_dir).readlines()]
    dataset = kitti_object(cfg_vis.DATA_DIR, split="testing", pred_dir=cfg_vis.predict_path,proposal_dir=cfg_vis.proposal_path)

    # for i in range(len(val_sample_idx)):
    #     if i<0:
    #         continue
    print("sample val id:", val_sample_idx[i])
    image_path = os.path.join(cfg_vis.DATA_DIR, "training", "image_2", val_sample_idx[i] + ".png")
    # img = cv2.imread(image_path)
    data_id = int(val_sample_idx[i])
    img = dataset.get_image(data_id)
    points = dataset.get_lidar(data_id)
    if show_raw==True:
        objects = None
        objects_pred = None
    else:
        # objects = dataset.get_label_objects(data_id)
        objects = None
        objects_pred = dataset.get_pred_objects(data_id)
        box_proposals = dataset.get_proposals(data_id)

    calib = dataset.get_calibration(data_id)
    cv2.imshow(str("fov_image"), img)

    utils.show_image_with_boxes(img, objects, calib, pred_object=objects_pred)
    if show_proposals:
        box_proposals_part_a = box_proposals[:100]
        box_proposals_cat = box_proposals_part_a+box_proposals[::50]
        utils.show_lidar_with_boxes(points, objects, calib, objects_pred=box_proposals_cat)
    else:
        utils.show_lidar_with_boxes(points, objects, calib, objects_pred=objects_pred)
    # lines = sys.stdin.readlines()
    # cv2.waitKey(0)

def save_gt_lidar():
    gt_annos_dict_path = cfg_vis.KITTI_TRAIN_VAL_INFO_PATH
    with open(gt_annos_dict_path, "rb") as f:
        gt_annos_dict = pickle.load(f)
    index = []
    gt_lidar_box_list = []
    for i, det in enumerate(gt_annos_dict):
        index.append(int(det["image"]["image_id"]))
        gt_lidar_box_list.append(det["annos"]["gt_boxes_lidar"])
    real_index = np.argsort(index)
    gt_lidar_box_list = [gt_lidar_box_list[i] for i in real_index]
    gt_lidar_boxs_path = os.path.join(cfg_vis.KITTI_INFO_DIR, "lida_gt_list.pkl")
    with open(gt_lidar_boxs_path, "wb") as f:
        pickle.dump(gt_lidar_box_list, f)
    print("save success!")

def read_bin(path):
    path = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/rosbag/leishen/2020-12-04-17-00-15-bin/5577.bin"
    points = np.fromfile(path,dtype=np.float32).reshape(-1, 4)
    return points

def render_points(points):
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
    )
    fig = utils.draw_lidar(points, fig=fig)
    mlab.show()


if __name__ == '__main__':
    aug_data_path = cfg_vis.AUGMENTATION_DATA_PATH
    raw_data_path = cfg_vis.RAW_DATA_PATH
    image_path = cfg_vis.IMAGE_PATH
    pred_path = cfg_vis.predict_path
    gt_annos_dict_path = cfg_vis.KITTI_TRAIN_VAL_INFO_PATH
    gt_lidar_path = cfg_vis.gt_lidar_path

    val_split_dir = cfg_vis.val_dataset
    val_sample_idx = [x.strip() for x in open(val_split_dir).readlines()]
    # if not os.path.exists(aug_data_path) or not os.path.exists(raw_data_path):
    #     print("generating data for vis.......")
    #     from vis_points.save_points_for_vis import generate_data_for_vis
    #     generate_data_for_vis(split=cfg_vis.SPLIT,sample_idx=cfg_vis.SAMPLE_INDEX)
    #     print("generate complete !")
    if cfg_vis.SHOW == "raw_points":
        cfg_vis.val_idx = 211
        for i in range(len(val_sample_idx)):
            if i!=cfg_vis.val_idx:
                continue
            print(i)
            id = val_sample_idx[i]
            point_file = id + ".bin"
            lidar_path = os.path.join(cfg_vis.KITTI_DIR, "velodyne", point_file)
            image_name = id + ".png"
            image_path = os.path.join(cfg_vis.KITTI_DIR, "image_2", image_name)
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            fig = mlab.figure(
                figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
            )
            fig = utils.draw_lidar(points, fig=fig)

            image_file = cv2.imread(image_path)
            cv2.imshow("image", image_file)
            mlab.show()

    if cfg_vis.SHOW == "center_points":
        split = "training"
        point_file = cfg_vis.SAMPLE_INDEX + ".bin"
        lidar_path = os.path.join(cfg_vis.KITTI_DIR, point_file)
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        gt_lidar_box = get_labels(cfg_vis.SAMPLE_INDEX, cfg_vis.DATA_DIR, split)
        gt_boxes_corners = boxes3d_to_corners3d_lidar(gt_lidar_box[:, :7])
        points_in_boxes, points_out_boxes = utils.points_indices(points, gt_lidar_box)
        points_size = np.array([3]).repeat(points_in_boxes.shape[0], 0)
        #读取检测结果
        with open(pred_path,"rb") as f:
            pred_dicts = pickle.load(f)
        for det_anno in pred_dicts:
            if det_anno['frame_id']==cfg_vis.SAMPLE_INDEX:
                det_gt = det_anno
                break
        det_boxes = det_gt['boxes_lidar']
        det_boxes_corners = boxes3d_to_corners3d_lidar(det_boxes[:, :7])
        class_name = list(det_gt["name"])
        fig = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
        )
        plot_forgroud_points = True#是否显示前景点
        if plot_forgroud_points == True:
            fig = utils.draw_lidar(points_out_boxes, fig=fig)
            mlab.points3d(points_in_boxes[:, 0],
                      points_in_boxes[:, 1],
                      points_in_boxes[:, 2],
                      points_size,
                      color=(1, 0, 0),
                      mode="sphere",
                      scale_factor=0.05,
                      figure=fig)
        elif plot_forgroud_points == False:
            pass
        show_raw_points = False#是否显示原始点云
        if show_raw_points==True:
            fig = utils.draw_lidar(points, fig=fig)
        show_gt_box = False#是否显示标签Box
        if show_gt_box == True:
            fig = utils.draw_gt_boxes3d(gt_boxes_corners, color=(1, 0, 0), fig=fig)
        show_box = False#是否显示Box
        if show_box == True:
            show_class = False
            if show_class!=False:
                fig = utils.draw_gt_boxes3d(det_boxes_corners, color=(0, 1, 0),label=class_name,draw_text=True, fig=fig)
            elif show_class==False:
                fig = utils.draw_gt_boxes3d(det_boxes_corners, color=(0, 1, 0),draw_text=False, fig=fig)
        elif show_box== False:
            pass
        FPS_available= False#是否显示采样之后的点
        if FPS_available==True:
            keypoints = fps_d(points[np.newaxis,...],2048)
            points_in_boxes, points_out_boxes = utils.points_indices(keypoints, gt_lidar_box)
            plot_seg_in_keypoints = False
            if plot_seg_in_keypoints ==True:
                fig = utils.draw_lidar(points_out_boxes, fig=fig)
                points_size = np.array([3]).repeat(points_in_boxes.shape[0], 0)
                mlab.points3d(points_in_boxes[:, 0],
                          points_in_boxes[:, 1],
                          points_in_boxes[:, 2],
                          points_size,
                          color=(1, 0, 0),
                          mode="sphere",
                          scale_factor=0.2,
                          figure=fig)
            elif plot_seg_in_keypoints==False:
                fig = utils.draw_lidar(keypoints, fig=fig)
        show_center = False#是否显示中心点
        if show_center==True:
            points_size = np.array([3]).repeat(gt_lidar_box.shape[0], 0)
            mlab.points3d(gt_lidar_box[:, 0],
                          gt_lidar_box[:, 1],
                          gt_lidar_box[:, 2],
                          points_size,
                          color=(1, 0, 0),
                          mode="sphere",
                          scale_factor=0.15,
                          figure=fig)
        image_file = cv2.imread(image_path)
        cv2.imshow(cfg_vis.SAMPLE_INDEX, image_file)
        mlab.show()


    if cfg_vis.SHOW == "show_raw_points_with_boxes":
        with open(raw_data_path, "rb") as f:
            raw_data = pickle.load(f)
        points = raw_data["points"]
        gt_boxes = raw_data["gt_boxes"]

        gt_boxes_corners = boxes3d_to_corners3d_lidar(gt_boxes[:,:7])
        points_in_boxes,points_out_boxes = utils.points_indices(points,gt_boxes)
        points_size = np.array([3]).repeat(points_in_boxes.shape[0], 0)

        fig = mlab.figure(
            figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000)
        )
        fig = utils.draw_lidar(points_out_boxes,fig=fig)

        mlab.points3d(points_in_boxes[:,0],
                      points_in_boxes[:,1],
                      points_in_boxes[:,2],
                      points_size,
                      color = (1,1,1),
                      mode = "sphere",
                      scale_factor = 0.05,
                      figure = fig)
        fig = utils.draw_gt_boxes3d(gt_boxes_corners,color=(1,0,0),fig=fig)
        image_file = cv2.imread(image_path)
        cv2.imshow(cfg_vis.SAMPLE_INDEX,image_file)
        mlab.show()
    elif  cfg_vis.SHOW == "show_aug_points_with_boxes":
        with open(aug_data_path, "rb") as f:
            aug_data = pickle.load(f)
        points = aug_data["points"]
        gt_boxes = aug_data["gt_boxes"]
        gt_boxes_corners = boxes3d_to_corners3d_lidar(gt_boxes[:, :7])
        points_in_boxes, points_out_boxes = utils.points_indices(points, gt_boxes)
        points_size = np.array([3]).repeat(points_in_boxes.shape[0], 0)

        fig = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
        )
        fig = utils.draw_lidar(points_out_boxes, fig=fig)

        mlab.points3d(points_in_boxes[:, 0],
                      points_in_boxes[:, 1],
                      points_in_boxes[:, 2],
                      points_size,
                      color=(1, 1, 1),
                      mode="sphere",
                      scale_factor=0.05,
                      figure=fig)
        fig = utils.draw_gt_boxes3d(gt_boxes_corners, color=(1, 0, 0), fig=fig)
        # image_file = cv2.imread(image_path)
        # cv2.imshow(cfg_vis.SAMPLE_INDEX, image_file)
        mlab.show()
    if cfg_vis.SHOW == "show_preds_v0":
        with open(gt_lidar_path, "rb") as f:
            gt_lidar_list = pickle.load(f)
        sample_idx_str = val_sample_idx[cfg_vis.val_idx]
        sample_idx = int(sample_idx_str)
        points_path = os.path.join(cfg_vis.KITTI_DIR, sample_idx_str + ".bin")
        point_xyzi = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
        with open(pred_path,"rb") as f:
            pred_dicts = pickle.load(f)

        for det_anno in pred_dicts:
            if det_anno['sample_idx'][0]==sample_idx:
                det_gt = det_anno
                break
        det_boxes = det_gt['boxes_lidar']
        gt_boxes = gt_lidar_list[sample_idx]
        gt_boxes_corners = boxes3d_to_corners3d_lidar(gt_boxes[:, :7])
        det_boxes_corners = boxes3d_to_corners3d_lidar(det_boxes[:, :7])
        points_in_boxes, points_out_boxes = utils.points_indices(point_xyzi, det_boxes)
        fig = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
        )
        fig = utils.draw_lidar(points_out_boxes, fig=fig)
        points_size = np.array([3]).repeat(points_in_boxes.shape[0], 0)
        mlab.points3d(points_in_boxes[:, 0],
                      points_in_boxes[:, 1],
                      points_in_boxes[:, 2],
                      points_size,
                      color=(1, 1, 1),
                      mode="sphere",
                      scale_factor=0.05,
                      figure=fig)
        fig = utils.draw_gt_boxes3d(gt_boxes_corners, color=(1, 0, 0), fig=fig)
        fig = utils.draw_gt_boxes3d(det_boxes_corners, color=(0, 1, 0), fig=fig)
        mlab.show()

    if cfg_vis.SHOW == "show_preds_v1":
        #save_gt_lidar()

        with open(gt_lidar_path,"rb") as f:
            gt_lidar_list = pickle.load(f)
        sample_idx_str = val_sample_idx[cfg_vis.val_idx]
        sample_idx = int(sample_idx_str)


        points_path = os.path.join(cfg_vis.KITTI_DIR, sample_idx_str + ".bin")

        point_xyzi = np.fromfile(points_path,dtype=np.float32).reshape(-1,4)
        draw_fps = False

        if draw_fps == True:
            import pvdet.model.pointnet2.pointnet2_stack.pointnet2_utils as pointnet2_stack_utils
            import torch
            nsample_list = [4096,512]
            sampled_points = torch.tensor(point_xyzi[:, 0:3]).unsqueeze(dim=0).cuda()
            for i,nsample in enumerate(nsample_list):
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), nsample).long()
                if sampled_points.shape[1] < nsample:
                    empty_num = nsample - sampled_points.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num].clone()

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
                sampled_points = keypoints

            keypoints_numpy = keypoints.squeeze(dim=0).cpu().numpy()

        with open(pred_path,"rb") as f:
            pred_dicts = pickle.load(f)
        for det_anno in pred_dicts:
            if det_anno['frame_id']==sample_idx_str:
                det_gt = det_anno
                break
        if det_gt is None:
            raise ValueError
        det_boxes = det_gt['boxes_lidar']
        class_name = list(det_gt["name"])
        gt_boxes = gt_lidar_list[sample_idx]

        gt_boxes_corners = boxes3d_to_corners3d_lidar(gt_boxes[:, :7])
        det_boxes_corners = boxes3d_to_corners3d_lidar(det_boxes[:, :7])

        points_in_boxes, points_out_boxes = utils.points_indices(point_xyzi[:,0:3], gt_boxes)
        fig = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
        )
        fig = utils.draw_gt_boxes3d(gt_boxes_corners, color=(1, 0, 0),fig=fig)
        fig = utils.draw_gt_boxes3d(det_boxes_corners, color=(0, 1, 0),label=class_name,draw_text=True, fig=fig)
        # fig = utils.draw_gt_boxes3d(det_boxes_corners, color=(0, 1, 0), fig=fig)
        fig = utils.draw_lidar(points_out_boxes, fig=fig)

        points_size = np.array([3]).repeat(points_in_boxes.shape[0], 0)
        fig = mlab.points3d(points_in_boxes[:, 0],
                            points_in_boxes[:, 1],
                            points_in_boxes[:, 2],
                            points_size,
                            color=(1, 0, 0),
                            mode="sphere",
                            scale_factor=0.08,
                            figure=fig)

        drow_center = False

        if drow_center == True:
            x = np.array([gt_boxes_corners[n][:,0].sum()/8 for n in range(len(gt_boxes))])
            y = np.array([gt_boxes_corners[n][:,1].sum()/8 for n in range(len(gt_boxes))])
            z = np.array([gt_boxes_corners[n][:,2].sum()/8 for n in range(len(gt_boxes))])
            mlab.points3d(x,y,z,color=(1, 0, 0),mode="sphere",scale_factor=0.5,figure = fig)

        #fig = utils.draw_lidar(keypoints_numpy,fig=fig)
        mlab.show()


    if cfg_vis.SHOW == "single_stage_model_results":
        val_split_dir = cfg_vis.val_dataset
        val_sample_idx = [x.strip() for x in open(val_split_dir).readlines()]
        for i in range(len(val_sample_idx)):
            if i!=2800:
                continue
            print("sample val id:",i)
            show_predicted_results(i)


    elif cfg_vis.SHOW == "show_img_with_box":
        val_sample_idx = [x.strip() for x in open(val_split_dir).readlines()]
        for i in range(len(val_sample_idx)):
            show_img_with_box(i)
        # input_str = input()
        # print(input_str)
        # mlab.clf()
        # if input_str == "kill":
        #     print("done")




