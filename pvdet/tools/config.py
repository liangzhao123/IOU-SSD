from easydict import EasyDict as edict
import os

cfg=edict()
cfg.print_info = False
cfg.CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']
cfg.TAG = "pvrcnn" #PartA2
using_remote = False

if using_remote:
    cfg.DATA_DIR = "/home/ubuntu-502/liang/kitti"
    cfg.CODE_DIR = "/home/ubuntu-502/liang/PVRCNN-V1.1"
else:
    cfg.DATA_DIR = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/kitti"
    cfg.CODE_DIR = "/home/liang/for_ubuntu502/PVRCNN-V1.1"

cfg.TIME_FILE = os.path.join(cfg.CODE_DIR,"output","figures")
cfg.DATA_CONFIG = edict()
cfg.DATA_CONFIG.FOV_POINTS_ONLY=True
cfg.DATA_CONFIG.NUM_WORKERS = 15
cfg.DATA_CONFIG.NUM_POINT_FEATURES={"total":4,"use":4}
cfg.DATA_CONFIG.VOXEL_GENERATOR={"MAX_POINTS_PER_VOXEL":5,
                                 "VOXEL_SIZE":[0.05,0.05,0.1]}
cfg.DATA_CONFIG.POINT_CLOUD_RANGE = [0,-40.0,-3.0,70.4,40.0,1.0]
cfg.DATA_CONFIG.MASK_POINTS_BY_RANGE=True
cfg.DATA_CONFIG.TRAIN = {"MAX_NUMBER_OF_VOXELS":16000,
                         "SHUFFLE_POINTS":True,
                         "INFO_PATH":["/home/ubuntu-502/liang/PVRCNN-V1.1/data/kitti/database_information/kitti_infos_train.pkl",
                                      "/home/ubuntu-502/liang/PVRCNN-V1.1/data/kitti/database_information/kitti_infos_val.pkl"] }#"/home/liang/PVRCNN-V0/database_information/kitti_infos_val.pkl"
cfg.DATA_CONFIG.TEST = {
    "SHUFFLE_POINTS":False,
    "MAX_NUMBER_OF_VOXELS":40000,
    "INFO_PATH":["/home/ubuntu-502/liang/PVRCNN-V1.1/data/kitti/database_information/kitti_infos_val.pkl"]
}
cfg.DATA_CONFIG.OFFICE_TEST_INFO_PATH =\
    ["/home/ubuntu-502/liang/PVRCNN-V1.1/data/kitti/database_information/kitti_infos_test.pkl"]


cfg.DATA_CONFIG.AUGMENTATION= edict({})
cfg.DATA_CONFIG.AUGMENTATION.DB_SAMPLER={"ENABLE":True,
                                         "DB_INFO_PATH":
                                             ["/home/ubuntu-502/liang/PVRCNN-V1.1/data/kitti/database_information/kitti_dbinfos_train.pkl",
                                              "/home/ubuntu-502/liang/PVRCNN-V1.1/data/kitti/database_information/kitti_dbinfos_trainval.pkl"],
                                         "PREPARE":{"filter_by_difficulty":[-1],
                                                    "filter_by_min_points":['Car:10', 'Pedestrian:10', 'Cyclist:10']},
                                         "RATE":1.0,
                                         "SAMPLE_GROUPS":['Car:15', 'Pedestrian:10', 'Cyclist:10'],
                                         'USE_ROAD_PLANE': True}
cfg.DATA_CONFIG.AUGMENTATION.NOISE_PER_OBJECT = {"ENABLED":True,
                                                 "GT_LOC_NOISE_STD":[1.0, 1.0, 0.5],
                                                 "GT_ROT_UNIFORM_NOISE":[-0.78539816, 0.78539816]}
cfg.DATA_CONFIG.AUGMENTATION.NOISE_GLOBAL_SCENE={"ENABLED":True,
                                                  "GLOBAL_ROT_UNIFORM_NOISE":[-0.78539816, 0.78539816],
                                                  "GLOBAL_SCALING_UNIFORM_NOISE":[0.95,1.05]}





cfg.MODEL = edict()
cfg.MODEL.VFE = edict()
cfg.MODEL.VFE = {"NAME":"MeanVoxelFeatureExtractor",
                 "ARGS": {}}



cfg.MODEL.RPN = edict()
cfg.MODEL.RPN = {"PARAMS_FIXED":False,
                 "BACKBONE":{"NAME": "UNetV2",
                             "TARGET_CONFIG":{"GENERATED_ON":"dataset",
                                               "GT_EXTEND_WIDTH":0.2,
                                               "MEAN_SIZE":{"Car":[1.6,3.9,1.56],
                                                "Pedestrian":[0.6,0.8,1.73],
                                                "Cyclist":[0.6,1.76,1.73]}
                                       },
                             "SEG_MASK_SCORE_THRESH":0.3,
                             "ARGS":{}
                             },
                 "RPN_HEAD":{
                             "num_bev_features": 256,

                             "TARGET_CONFIG":{
                                "num_anchors_per_location":2,
                                 "ANCHOR_GENERATOR":[
                                     {"anchor_range":[0,-40,-1.78, 70.4,40,-1.78],
                                      "anchor_bottom_heights":[-1.78],
                                      "align_center":False,
                                        "class_name":"Car",
                                        "matched_threshold":0.6,
                                        "rotations":[0,1.57],
                                        "sizes":[[1.6,3.9,1.56]],
                                        'feature_map_stride': 8,
                                        "unmatched_threshold":0.45
                                        },
                                        {"anchor_range":[0,-40,-0.6, 70.4,40,-0.6],
                                         "anchor_bottom_heights": [-0.6],
                                         "align_center": False,
                                        "class_name":"Pedestrian",
                                         "matched_threshold":0.5,
                                         'feature_map_stride': 8,
                                         "rotations": [0, 1.57],
                                        "sizes": [[0.6, 0.8, 1.73]],
                                            "unmatched_threshold": 0.35
                                            },
                                        {"anchor_range":[0,-40,-0.6, 70.4,40,-0.6],
                                         "anchor_bottom_heights": [-0.6],
                                         "align_center": False,
                                        "class_name":"Cyclist",
                                            "matched_threshold":0.5,
                                         'feature_map_stride': 8,
                                            "rotations": [0, 1.57],
                                        "sizes": [[0.6, 1.76, 1.73]],
                                        "unmatched_threshold": 0.35
                                        }
                                                                  ],
                                 "Assigner_Targets_Config":{
                                        "BOX_CODER":"ResidualCoder_v1",
                                        "REGION_SIMILARITY_FN":"nearest_iou_similarity",
                                        "POS_FRACTION":-1.0,
                                        "SAMPLE_SIZE":512,
                                        "DOWNSAMPLED_FACTOR":8,
                                        "NORM_BY_NUM_EXAMPLES":False,
                                        "MATCH_HEIGHT":False},
                                              },
                             "ARGS":{"use_norm":True,
                                     "concat_input":False,
                                     "num_input_features":256,
                                     "layer_nums":[5,5],
                                     "layer_strides":[1,2],
                                     "num_filters":[128,256],
                                     "upsample_strides":[1,2],
                                     "num_upsample_filters":[256,256],
                                     "encode_background_as_zeros":True,

                                     "use_direction_classifier":True,
                                     "num_direction_bins":2,
                                     "dir_offset":0.78539,
                                     "dir_limit_offset":0.0,
                                     "use_binary_dir_classifier":False},
                            "LOSSES":{
                                "cls_weight":1.0,
                                "reg_loss_weight":2.0,
                                "dir_loss_weight":0.2,
                                "code_loss_weight":[1.0,1.0,1.0,
                                                    1.0,1.0,1.0,
                                                    1.0]},



                             }
                 }
cfg.MODEL.VOXEL_SA = {
    "ENABLE":True,
    "POINT_SOURCE": "raw_points",
    "NUM_KEYPOINTS": 2048,
    "NUM_OUTPUT_FEATURES": 128,
    "SAMPLE_METHOD": "FPS",

    "FEATURES_SOURCE": ['bev', 'x_conv1', 'x_conv2', 'x_conv3', 'x_conv4', 'raw_points'],
    "SA_LAYER":
            {"raw_points":
                 {"MLPS":[[16, 16], [16, 16]],
                "POOL_RADIUS": [0.4, 0.8],
                "NSAMPLE": [16, 16]},
            "x_conv1":{
                "DOWNSAMPLE_FACTOR": 1,
                "MLPS": [[16, 16], [16, 16]],
                "POOL_RADIUS": [0.4, 0.8],
                "NSAMPLE": [16, 16]},
            "x_conv2":
                 {"DOWNSAMPLE_FACTOR": 2,
                "MLPS": [[32, 32], [32, 32]],
                "POOL_RADIUS": [0.8, 1.2],
                "NSAMPLE": [16, 32]},
            "x_conv3":
                 {"DOWNSAMPLE_FACTOR": 4,
                  "MLPS": [[64, 64], [64, 64]],
                  "POOL_RADIUS": [1.2, 2.4],
                  "NSAMPLE": [16, 32], },
            "x_conv4":
                 {"DOWNSAMPLE_FACTOR": 8,
                  "MLPS": [[64, 64], [64, 64]],
                  "POOL_RADIUS": [2.4, 4.8],
                  "NSAMPLE": [16, 32],}
            }
}
cfg.MODEL.POINT_HEAD={
            "NAME": "PointHeadSimple",
            "CLS_FC": [256, 256],
            "CLASS_AGNOSTIC": True,
            "USE_POINT_FEATURES_BEFORE_FUSION": True,
            "TARGET_CONFIG":{
                "GT_EXTRA_WIDTH": [0.2, 0.2, 0.2],},
            "LOSS_CONFIG":
                    {"LOSS_REG": "smooth-l1",
                    "LOSS_WEIGHTS": {
                        'point_cls_weight': 1.0,},
                     }
            }




cfg.MODEL.PV_RCNN = {
        "CLASS_AGNOSTIC": True,
        "SHARED_FC": [256, 256],
        "CLS_FC": [256, 256],
        "REG_FC": [256, 256],
        "DP_RATIO": 0.3,
        "NMS_CONFIG":
            {"TRAIN":
                 {"NMS_TYPE": "nms_gpu",
                "MULTI_CLASSES_NMS": False,
                "NMS_PRE_MAXSIZE": 9000,
                "NMS_POST_MAXSIZE": 512,
                "NMS_THRESH": 0.8},
            "TEST":
                {"NMS_TYPE": "nms_gpu",
                "MULTI_CLASSES_NMS": False,
                "NMS_PRE_MAXSIZE": 1024,
                "NMS_POST_MAXSIZE": 40,
                "NMS_THRESH": 0.7}
             },

        "ROI_GRID_POOL":{
            "GRID_SIZE": 6,
            "MLPS": [[64, 64], [64, 64]],
            "POOL_RADIUS": [0.8, 1.6],
            "NSAMPLE": [16, 16],
            "POOL_METHOD": "max_pool",
        },

        "TARGET_CONFIG":{
            "BOX_CODER": "ResidualCoder_v1",
            "ROI_PER_IMAGE": 128,
            "FG_RATIO": 0.5,

            "SAMPLE_ROI_BY_EACH_CLASS": True,
            "CLS_SCORE_TYPE": "roi_iou",

            "CLS_FG_THRESH": 0.75,
            "CLS_BG_THRESH": 0.25,
            "CLS_BG_THRESH_LO": 0.1,
            "HARD_BG_RATIO": 0.8,

            "REG_FG_THRESH": 0.55,
        },
        "LOSS_CONFIG":{
            "CLS_LOSS": "BinaryCrossEntropy",
            "REG_LOSS": "smooth-l1",
            "CORNER_LOSS_REGULARIZATION": True,
            "LOSS_WEIGHTS": {
                'rcnn_cls_weight': 1.0,
                'rcnn_reg_weight': 1.0,
                'rcnn_corner_weight': 1.0,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }
        }
}
cfg.MODEL.POST_PROCESSING=edict(
    {
        "RECALL_THRESH_LIST": [0.3, 0.5, 0.7],
        "SCORE_THRESH": 0.1,
        "OUTPUT_RAW_SCORE": False,
        "EVAL_METRIC": "kitti",
        "NMS_CONFIG":
            {"MULTI_CLASSES_NMS": False,
            "NMS_TYPE": "nms_gpu",
            "NMS_THRESH": 0.1,
            "NMS_PRE_MAXSIZE": 4096,
            "NMS_POST_MAXSIZE": 500,}
    }
)
cfg.MODEL.OPTIMIZATION = edict(

    {               "OPTIMIZER":"adam_onecycle",
                     "LR":0.01,
                     "WEIGHT_DECAY":0.01,
                     "MOMENTUM":0.9,
                     "MOMS":[0.95,0.85],
                     "PCT_START":0.4,
                     "DIV_FACTOR":10,
                     "DECAY_STEP_LIST":[40,60,70],
                     "LR_DECAY":0.1,
                     "LR_CLIP":0.0000001,
                     "LR_WARMUP":False,
                     "WARMUP_EPOCH":1,
                     "GRAD_NORM_CLIP":10}

)




#这部分是train的args参数
cfg.train_args = edict()
cfg.train_args={"BATCH_SIZE" : 2,
                "pretrained_model":None,
                "LOCAL_RANK":0,
                "ckpt":"/home/liang/PVRCNN-V0/output/ckpt/",
                "ckpt_interval":1,
                "max_ckpt_save_num":50,
                "epoch":80}


#产生数据增强中的gt database的info路径和数据路径
cfg.GT_DATABASE_INFO="/home/liang/PVRCNN-V0/database_information"
cfg.GT_DATABASE = "/home/liang/PVRCNN-V0/database"
cfg.KITTI_INFO_PATH = "/home/liang/PVRCNN-V0/database_information"



#关于test的参数也可以在这里修该
cfg.TEST_ARGS =edict()
cfg.TEST_ARGS = {
    "ckpt":"/home/liang/PVRCNN-V1/ckpt",
    "ckpt_file":None,
    "EXTRA_TAG":"default",#for experiment
    "eval_all":False,#是否使用最终的模型进行eval
    "ckpt_id":"17",#预测哪个ckpt的模型
    "eval_tag":"default",

    "batch_size":4,

    "start_epoch":0,
    "max_wait_mins":30,

    "save_to_file":True,
}
cfg.DEVICE = 0



def log_cfg_to_file(cfg,pre="cfg",logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key],edict):
            logger.info("\n%s.%s = edict()"%(pre,key))
            log_cfg_to_file(cfg[key],pre=pre+"."+key,logger=logger)
            continue
        logger.info("%s.%s.%s" % (pre,key,val))









if __name__=="__main__":
    print(cfg.MODEL.RPN.BACKBONE.ARGS )

# {"raw_points":
#              {"MLPS":[[16, 16], [16, 16]],
#             "POOL_RADIUS": [0.4, 0.8],
#             "NSAMPLE": [16, 16]},
#         "x_conv1":{
#             "DOWNSAMPLE_FACTOR": 1,
#             "MLPS": [[16, 16], [16, 16]],
#             "POOL_RADIUS": [0.4, 0.8],
#             "NSAMPLE": [16, 16]},
#         "x_conv2":
#              {"DOWNSAMPLE_FACTOR": 2,
#             "MLPS": [[32, 32], [32, 32]],
#             "POOL_RADIUS": [0.8, 1.2],
#             "NSAMPLE": [16, 32]},
#         "x_conv3":
#              {"DOWNSAMPLE_FACTOR": 4,
#               "MLPS": [[64, 64], [64, 64]],
#               "POOL_RADIUS": [1.2, 2.4],
#               "NSAMPLE": [16, 32], },
#         "x_conv4":
#              {"DOWNSAMPLE_FACTOR": 8,
#               "MLPS": [[64, 64], [64, 64]],
#               "POOL_RADIUS": [2.4, 4.8],
#               "NSAMPLE": [16, 32],}
#         }