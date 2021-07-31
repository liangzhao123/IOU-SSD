from easydict import EasyDict as edict
import os

cfg = edict()


cfg_vis = edict()
cfg_vis.SHOW = "show_img_with_box"
#"show_aug_points_with_boxes" ,show_raw_points_with_boxes,raw_points
# show_preds_v1,show_preds_v0，show_preds_v2,single_stage_model_results,center_points
#important single_stage_model_results,show_img_with_box
cfg_vis.SAMPLE_INDEX = "000567"
cfg_vis.SPLIT = "val"
cfg_vis.CODE_DIR="/home/liang/PVRCNN-V0/"
cfg_vis.KITTI_INFO_DIR = "/home/liang/PVRCNN-V0/database_information/"
cfg_vis.KITTI_TRAIN_INFO_PATH = os.path.join(cfg_vis.KITTI_INFO_DIR,"kitti_infos_train.pkl")
cfg_vis.KITTI_VAL_INFO_PATH = os.path.join(cfg_vis.KITTI_INFO_DIR,"kitti_infos_val.pkl")
cfg_vis.KITTI_TRAIN_VAL_INFO_PATH = os.path.join(cfg_vis.KITTI_INFO_DIR,"kitti_infos_trainval.pkl")
cfg_vis.KITTI_TEST_INFO_PATH = os.path.join(cfg_vis.KITTI_INFO_DIR,"kitti_infos_test.pkl")
cfg_vis.AUGMENTATION_DATA_PATH = os.path.join(cfg_vis.CODE_DIR,"debug_file","process_%s.pth" % cfg_vis.SAMPLE_INDEX)
cfg_vis.RAW_DATA_PATH = os.path.join(cfg_vis.CODE_DIR,"debug_file","raw_%s.pth" % cfg_vis.SAMPLE_INDEX)
cfg_vis.IMAGE_PATH = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/kitti/%s/image_2/%s.png" % ("training" if cfg_vis.SPLIT != "test" else "testing" ,cfg_vis.SAMPLE_INDEX)
##"/home/liang/PVRCNN-V1.1-laptop/preds/v2/result.pkl",
# "/home/liang/PVRCNN-V1.1/output/PVAnet/fix_iou3d_bug/eval/epoch_80/result.pkl"
#"/home/liang/PVRCNN-V1.1/preds/laptop/result.pkl"


###########for single stage model
cfg.CODE_DIR = "/home/liang/for_ubuntu502/PVRCNN-V1.1"
cfg_vis.gt_lidar_path =os.path.join(cfg_vis.KITTI_INFO_DIR, "lida_gt_list.pkl")
cfg_vis.kitti_val_info= os.path.join(cfg.CODE_DIR, "single_stage_model",
                                                   "data/kitti/database_information/kitti_infos_val.pkl")
cfg_vis.kitti_gt_dir = os.path.join(cfg.CODE_DIR,"single_stage_model","data/kitti")
#TODO
cfg_vis.KITTI_DIR = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/kitti/training"
cfg_vis.DATA_DIR = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/kitti"
cfg_vis.val_dataset = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/kitti/ImageSets/test.txt"
# cfg_vis.points_path = os.path.join(cfg_vis.KITTI_DIR,cfg_vis.SAMPLE_INDEX+".bin")
cfg_vis.val_idx = 211 #28，default 58,801


# cfg_vis.predict_path = "/home/liang/for_ubuntu502/PVRCNN-V1.1/output/single_stage_model/0.0.5.8/eval/epoch_80/result.pkl"
# cfg_vis.predict_path = "/home/liang/for_ubuntu502/PVRCNN-V1.1/output/LZnet/0.0.6/eval/epoch_80/result.pkl"
# cfg_vis.predict_path = "/home/liang/for_ubuntu502/PVRCNN-V1.1/output/single_stage_model/0.0.5.1/eval/epoch_80/final_result/data"
cfg_vis.predict_path = "/home/liang/for_ubuntu502/PVRCNN-V1.1/output/deformable-rcnn/data"
cfg_vis.proposal_path = "/home/liang/for_ubuntu502/PVRCNN-V1.1/output/single_stage_model/0.0.3.2/eval/epoch_50/final_result/proposals"