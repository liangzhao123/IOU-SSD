
import vis_points.utils as utils
import cv2
import mayavi.mlab as mlab
from single_stage_model.dataset.leishen_dataset import anno_utils
import numpy as np
import glob
import os
from single_stage_model.dataset.leishen_dataset.utils import *
import pickle
import torch
from single_stage_model.iou3d_nms import iou3d_nms_utils



def show_box_with_anno(ann_dir,bin_dir,image_dir):
    anno_list = glob.glob(os.path.join(ann_dir, "*.txt"))
    image_list = glob.glob(os.path.join(image_dir, "*.png"))
    image_list.sort()
    anno_list.sort()
    bin_list = glob.glob(os.path.join(bin_dir, "*.bin"))
    bin_list.sort()
    for  anno_path, i in zip(anno_list, range(len(bin_list))):
        id = anno_path.split("/")[-1]
        id = id.split(".")[0]
        print("frame id:%s" % id)
        if id != "000543":
            continue
        image_path = os.path.join(image_dir,id+".png")
        bin_path = os.path.join(bin_dir, id + ".bin")
        objects = get_objects_from_label(anno_path)
        gt_boxes = get_info(objects)
        print("gt_boxes:", gt_boxes)
        gt_boxes_corners, temp_corners = center2corner_leishen(gt_boxes[:, :7])
        points = read_bin(bin_path)

        pts = np.unique(points,axis=0).reshape(-1,4)

        img = cv2.imread(image_path)
        fig = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
        )

        fig = utils.draw_lidar(pts, fig=fig,datatype="leishen")
        fig = utils.draw_gt_boxes3d(gt_boxes_corners, color=(1, 0, 0), fig=fig, gt_boxes_center=gt_boxes)
        cv2.imshow("image",img)
        mlab.show()
    print("done")

def show_single_frame():
    data_dir = "/media/liang/Elements/rosbag/leishen_e70_32/dataset_image_pcd"
    bin_dir = os.path.join(data_dir,"bin")
    id = 552
    anno_path = os.path.join(data_dir, "annotation",str(id).zfill(6)+".txt")
    bin_path = os.path.join(bin_dir, str(id).zfill(6)+".bin")
    objects = get_objects_from_label(anno_path)
    gt_boxes = get_info(objects)

    print("gt_boxes:", gt_boxes)
    gt_boxes_corners, temp_corners = center2corner_leishen(gt_boxes[:, :7])
    points = read_bin(bin_path)
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
    )

    fig = utils.draw_lidar(points, fig=fig, datatype="leishen")
    fig = utils.draw_gt_boxes3d(gt_boxes_corners, color=(1, 0, 0), fig=fig, gt_boxes_center=gt_boxes)
    mlab.show()

def statistic_points_in_bin(bin_dir):
    bin_list = glob.glob(os.path.join(bin_dir, "*.bin"))
    bin_list.sort()
    for i in bin_list:
        points = np.fromfile(i,np.float32).reshape(-1,4)
        num = len(points)
        id = i.split("/")[-1].split(".")[0]
        if num <40000:
            print(id)
def show_label():
    ann_dir = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/rosbag/leishen_e70_32/dataset_image_pcd/annotation"
    bin_dir = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/rosbag/leishen_e70_32/dataset_image_pcd/bin_1"
    image_dir = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/rosbag/leishen_e70_32/dataset_image_pcd/image/sorted"
    # sum_box(ann_dir)
    # statistic_points_in_bin(bin_dir)
    show_box_with_anno(ann_dir, bin_dir, image_dir)
    # points = np.fromfile(os.path.join(bin_dir,"000549.bin"),dtype=np.float32).reshape(-1,4)
    # show_single_frame()
    print("done")

def show_frame(id = 1,show_augmatation=False,show_predict=False,eval_save_pth=False):
    image_dir = os.path.join(DATA_DIR,"image","sorted")
    image_path = os.path.join(image_dir,str(id).zfill(6) + ".png")
    image_data = cv2.imread(image_path)
    if show_augmatation:
        anno_path = os.path.join(DATA_DIR, "augmentation", str(id).zfill(6) + ".txt")
        bin_dir = os.path.join(DATA_DIR, "augmentation")
        bin_path = os.path.join(bin_dir, str(id).zfill(6) + ".bin")
    else:
        anno_path = os.path.join(DATA_DIR, "annotation", str(id).zfill(6) + ".txt")
        bin_dir = os.path.join(DATA_DIR, "bin")
        bin_path = os.path.join(bin_dir, str(id).zfill(6) + ".bin")
    if show_predict and not show_augmatation:
        # "/for_ubuntu502/PVRCNN-V1.1/output/single_stage_model/leishen_1.1/eval/epoch_80/final_result/data"
        pred_path = os.path.join(OUTPUT_DIR,"single_stage_model",
                                 "leishen_1.3","eval","epoch_80","final_result",
                                 "data",str(id).zfill(6) + ".txt")
        predict_objects = get_objects_from_label(pred_path)
        det_boxes = get_info(predict_objects, drop_bus=True)

    if eval_save_pth:
        result_dir = os.path.join(CODE_DIR, "output", "single_stage_model", "leishen_1.3", "eval", "epoch_80")
        reuslt_file_path = os.path.join(result_dir, "result.pkl")
        with open(reuslt_file_path, "rb") as f:
            det_annos = pickle.load(f)
        for sample in det_annos:
            sample["frame_id"] == str(id).zfill(6)
            break
        det_boxes_ = sample['boxes_lidar']


    objects = get_objects_from_label(anno_path)
    gt_boxes = get_info(objects)

    if show_predict:
        if len(gt_boxes) > 0:
            values = iou3d_nms_utils.boxes_iou3d_gpu(torch.from_numpy(gt_boxes[:, :7].astype(np.float32)).cuda(),
                                                     torch.from_numpy(
                                                         det_boxes[:, :7].astype(np.float32)).cuda()).cpu().numpy()


    # gt_boxes = gt_boxes[6:7,:]
    # det_boxes = det_boxes[8:9,:]
    if len(gt_boxes)>0:
        gt_boxes_corners, temp_corners = center2corner_leishen(gt_boxes[:, :7])
    if show_predict:
        if len(det_boxes)>0:
            det_boxes_corners, det_temp_corners = center2corner_leishen(det_boxes[:, :7])
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # points = anno_utils.remove_points_in_boxes3d_v0(points,gt_boxes[0:1,:7])
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
    )
    # mlab.points3d(
    #     points[:, 0],
    #     points[:, 1],
    #     points[:, 2],
    #     color=(1, 1, 1),
    #     mode="point",
    #     colormap="gnuplot",
    #     scale_factor=0.3,
    #     figure=fig,
    # )
    fig = utils.draw_lidar(points, fig=fig, datatype="leishen",
                           draw_axis=True,draw_square_region=False,
                           draw_fov=False)
    if len(gt_boxes) > 0:
        fig = utils.draw_gt_boxes3d(gt_boxes_corners, color=(1, 0, 0), fig=fig, gt_boxes_center=gt_boxes)
    if show_predict:
        fig = utils.draw_gt_boxes3d(det_boxes_corners, color=(0, 1, 0), fig=fig, gt_boxes_center=gt_boxes)
    cv2.imshow("image",image_data)
    mlab.show()

if __name__ == '__main__':
    # get_leishen_annotation_info(DATA_DIR+"/annotation")
    for i in range(5,580,5):
        if i<135:
            continue
        print(i)
        show_frame(i)