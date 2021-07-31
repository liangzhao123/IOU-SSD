from easydict import EasyDict as edict
import os


cfg=edict()
using_remote = False
if using_remote:
    cfg.DATA_DIR = "/media/ubuntu-502/pan1/liang/leishen_e70_32/dataset_image_pcd"
    cfg.CODE_DIR = "/media/ubuntu-502/pan1/liang/PVRCNN-V1.1"
else:
    cfg.DATA_DIR = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/rosbag/leishen_e70_32/dataset_image_pcd"
    cfg.CODE_DIR = "/home/liang/for_ubuntu502/PVRCNN-V1.1"

cfg.LEISHEN = "leishen"
cfg.CLASS_NAMES = ['Car', 'Pedestrian', 'Rider','Bus']
cfg.LOCAL_RANK = 0

cfg.data_config = edict()


cfg.data_config = {
    "anchor_mask_enable":False,
    "VoxelGenerator":{"voxel_size": [0.05, 0.05, 0.1],
                                  "max_point_per_voxel": 5},
    "TRAIN":{"max_number_of_voxel": 30000,
                         "info_path": [os.path.join(cfg.CODE_DIR, "single_stage_model",
                                                    "data/leishen/database_information/leishen_infos_train.pkl"),
                                       # os.path.join(cfg.CODE_DIR, "single_stage_model",
                                       #              "data/leishen/database_information/leishen_infos_trainval.pkl")
                                       ],
                         "SHUFFLE_POINTS": True},
    "TEST":{"max_number_of_voxel": 40000,
                        "info_path": [os.path.join(cfg.CODE_DIR, "single_stage_model",
                                                   "data/leishen/database_information/leishen_infos_val.pkl")],
                        "SHUFFLE_POINTS": False,
                        },
    "offical_test_info_path": [
        os.path.join(cfg.CODE_DIR, "single_stage_model", "data/leishen/database_information/leishen_infos_test.pkl")],
    "mask_point_by_range": True,
    "mask_repeat_points":True,
    "fov_points_only": True,
    "point_cloud_range": [-25, -30, -3, 25.4, 30, 1],
    "num_used_features": 4,
    "anchor_mask_config":{"anchor_area_threshold":1},
    "augmentation": {
        "db_sampler": {
            "enable": True,
            "db_info_path": [os.path.join(cfg.CODE_DIR, "single_stage_model",
                                          "data/leishen/database_information/leishen_dbinfos_train.pkl"),
                             os.path.join(cfg.CODE_DIR, "single_stage_model",
                                          "data/leishen/database_information/leishen_dbinfos_trainval.pkl"),
                             os.path.join(cfg.CODE_DIR, "single_stage_model",
                                          "data/leishen/database_information/leishen_dbinfos_train_80%.pkl")],
            "PREPARE": {"filter_by_difficulty": [-1],
                        "filter_by_min_points": ['Car:10', 'Pedestrian:10', 'Rider:10','Bus:10']},
            "RATE": 1.0,
            "SAMPLE_GROUPS": ['Car:15', 'Pedestrian:10', 'Rider:10','Bus:0'],
            'USE_ROAD_PLANE': False},
        "NOISE_PER_OBJECT": {"ENABLED": True,
                             "GT_LOC_NOISE_STD": [1.0, 1.0, 0.5],
                             "GT_ROT_UNIFORM_NOISE": [-0.78539816, 0.78539816]},
        "NOISE_GLOBAL_SCENE": {"ENABLED": True,
                               "GLOBAL_ROT_UNIFORM_NOISE": [-0.78539816, 0.78539816],
                               "GLOBAL_SCALING_UNIFORM_NOISE": [0.95, 1.05]}
    },

}

cfg.model = edict({
    "Conv3d":{"add_layers_for_conv3d":False},
    "detection_head": {
        "feature_map": [200, 176],
        "point_cloud_range": cfg.data_config.point_cloud_range,
        "dir_cls_bin": 2,
        "dir_offset": 0.785,
        "dir_limit_offset": 0,
        "num_class": len(cfg.CLASS_NAMES),
        "using_backgroud_as_zero": False,
        "using_iou_branch":False,
        "iou_bin_num":5,
        "target_config": {
            "match_height": False,
            "code_size": 7,
            "norm_by_num_samples": False,
            "pos_fraction":-1,
            "sample_size":512,
            "anchor_generator": [
                {"anchor_range": [-25, -30, -1.78, 25.4, 30, -1.78],
                 "anchor_bottom_heights": [-1.78],
                 "align_center": False,
                 "class_name": "Car",
                 "matched_threshold": 0.6,
                 "rotations": [0, 1.57],
                 "sizes": [[1.82, 4.12, 1.48]],
                 'feature_map_stride': 8,
                 "unmatched_threshold": 0.45
                 },
                {"anchor_range": [-25, -30, -0.6, 25.4, 30, -0.6],
                 "anchor_bottom_heights": [-0.6],
                 "align_center": False,
                 "class_name": "Pedestrian",
                 "matched_threshold": 0.5,
                 'feature_map_stride': 8,
                 "rotations": [0, 1.57],
                 "sizes": [[0.6, 0.8, 1.73]],
                 "unmatched_threshold": 0.35
                 },
                {"anchor_range": [-25, -30, -0.6, 25.4, 30, -0.6],
                 "anchor_bottom_heights": [-0.6],
                 "align_center": False,
                 "class_name": "Rider",
                 "matched_threshold": 0.5,
                 'feature_map_stride': 8,
                 "rotations": [0, 1.57],
                 "sizes": [[0.75, 1.45, 1.39]],
                 "unmatched_threshold": 0.35
                 },
                {"anchor_range": [-25, -30, -1.78, 25.4, 30, -1.78],
                 "anchor_bottom_heights": [-1.78],
                 "align_center": False,
                 "class_name": "Bus",
                 "matched_threshold": 0.6,
                 'feature_map_stride': 8,
                 "rotations": [0, 1.57],
                 "sizes": [[2.83, 9.48, 2.77]],
                 "unmatched_threshold": 0.45
                 }
            ],
        },
        "loss_config": {
            "cls_weight": 1.0,
            "reg_loss_weight": 2.0,
            "dir_loss_weight": 0.2,
            "code_loss_weight": [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0,
                                 1.0],
            # "iou_loss_weight":[1],
            # "iou_loss_bin_weight":0.1,
            # "iou_loss_residual_weight":0.1
        }
    },
    "IouHead":{
        "enable":True,
        "using_cross_entropy":True,
        "using_focal_loss":False,
        "selected_num":9000,
        "iou_bin_num":5,
        "iou_loss_weight":[1],
        "iou_loss_bin_weight":0.5,
        "iou_loss_residual_weight":0.5
    },
    #     "target_conf80ig":{
    #     "ANCHOR_GENERATOR":[
    #         {"anchor_range": [0, -40, -1.78, 70.4, 40, -1.78],
    #          "anchor_bottom_heights": [-1.78],
    #          "align_center": False,
    #          "class_name": "Car",
    #          "matched_threshold": 0.6,
    #          "rotations": [0, 1.57],
    #          "sizes": [[1.6, 3.9, 1.56]],
    #          'feature_map_stride': 8,
    #          "unmatched_threshold": 0.45
    #          },
    #         {"anchor_range": [0, -40, -0.6, 70.4, 40, -0.6],
    #          "anchor_bottom_heights": [-0.6],
    #          "align_center": False,
    #          "class_name": "Pedestrian",
    #          "matched_threshold": 0.5,
    #          'feature_map_stride': 8,
    #          "rotations": [0, 1.57],
    #          "sizes": [[0.6, 0.8, 1.73]],
    #          "unmatched_threshold": 0.35
    #          },
    #         {"anchor_range": [0, -40, -0.6, 70.4, 40, -0.6],
    #          "anchor_bottom_heights": [-0.6],
    #          "align_center": False,
    #          "class_name": "Cyclist",
    #          "matched_threshold": 0.5,
    #          'feature_map_stride': 8,
    #          "rotations": [0, 1.57],
    #          "sizes": [[0.6, 1.76, 1.73]],
    #          "unmatched_threshold": 0.35
    #          }
    #         ],
    #     "Assigner_Targets_Config": {
    #         "BOX_CODER": "ResidualCoder_v1",
    #         "REGION_SIMILARITY_FN": "nearest_iou_similarity",
    #         "POS_FRACTION": -1.0,
    #         "SAMPLE_SIZE": 512,
    #         "DOWNSAMPLED_FACTOR": 8,
    #         "NORM_BY_NUM_EXAMPLES": False,
    #         "MATCH_HEIGHT": False},
    # },


    "optimization": {"OPTIMIZER": "adam_onecycle",
                     "LR": 0.01,
                     "WEIGHT_DECAY": 0.01,
                     "MOMENTUM": 0.9,
                     "MOMS": [0.95, 0.85],
                     "PCT_START": 0.4,
                     "DIV_FACTOR": 10,
                     "DECAY_STEP_LIST": [40, 60, 70],
                     "LR_DECAY": 0.1,
                     "LR_CLIP": 0.0000001,
                     "LR_WARMUP": False,
                     "WARMUP_EPOCH": 1,
                     "GRAD_NORM_CLIP": 10},
    "post_processing": {
        "stratgy_name":["using_gt","using_iou","using_class_score","cls_iou_blend"],
        "stratgy_id":2,
        "cls_threshold": 0.1,
        "iou_thresh":0.1,
        "topk_iou_ratio":0.90,
        "pre_selection_num": 9000,
        "post_selected_num": 100,
        "nms_thresh": 0.0001,
        "recall_thresh_list": [0.3, 0.5, 0.7],
    },

})

cfg.leishen_info_path = os.path.join(cfg.CODE_DIR, "single_stage_model/data/leishen/database_information")
cfg.leishen_gt_database = os.path.join(cfg.CODE_DIR, "single_stage_model/data/leishen/database")





def log_cfg_to_file(cfg,pre="cfg",logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key],edict):
            logger.info("\n%s.%s = edict()"%(pre,key))
            log_cfg_to_file(cfg[key],pre=pre+"."+key,logger=logger)
            continue
        logger.info("%s.%s.%s" % (pre,key,val))
