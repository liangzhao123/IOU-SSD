from easydict import EasyDict as edict

import numpy as np
import os
from pathlib import Path


cfg=edict()

cfg.CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']
cfg.TAG = "PartA2_sprcn" #PartA2_Novel
cfg.DATA_DIR="/home/liang/kitti/"
cfg.CODE_DIR = "/home/liang/PVRCNN-V0/"
cfg.IMAGE_SHAPE= np.array([375, 1242],dtype=np.int32)

cfg.DATA_CONFIG = edict()
cfg.DATA_CONFIG.FOV_POINTS_ONLY=True
cfg.DATA_CONFIG.NUM_WORKERS = 15
cfg.DATA_CONFIG.NUM_POINT_FEATURES={"total":4,"use":4}
cfg.DATA_CONFIG.VOXEL_GENERATOR={"MAX_POINTS_PER_VOXEL":5,
                                 "VOXEL_SIZE":[0.05,0.05,0.1]}
cfg.DATA_CONFIG.POINT_CLOUD_RANGE = [0,-40,-3,70.4,40,1]
cfg.DATA_CONFIG.TRAIN = {"MAX_NUMBER_OF_VOXELS":16000,
                         "SHUFFLE_POINTS":True,
                         "INFO_PATH":["/home/liang/PVRCNN-V0/database_information/kitti_infos_train.pkl"] }#"/home/liang/PVRCNN-V0/database_information/kitti_infos_val.pkl"
cfg.DATA_CONFIG.TEST = {
    "SHUFFLE_POINTS":False,
    "MAX_NUMBER_OF_VOXELS":40000,
    "INFO_PATH":["/home/liang/PVRCNN-V0/database_information/kitti_infos_val.pkl"]
}
cfg.DATA_CONFIG.MASK_POINT_BY_RANGE=True
cfg.DATA_CONFIG.AUTMENTATION= edict({})
cfg.DATA_CONFIG.AUTMENTATION.DB_SAMPLER={"ENABLE":True,
                                         "DB_INFO_PATH":
                                             ["/home/liang/PVRCNN-V0/database_information/kitti_dbinfos_train.pkl"],
                                         "PREPARE":{"filter_by_difficulty":[-1],
                                                    "filter_by_min_points":['Car:10', 'Pedestrian:10', 'Cyclist:10']},
                                         "RATE":1.0,
                                         "SAMPLE_GROUPS":['Car:15', 'Pedestrian:10', 'Cyclist:10'],
                                         'USE_ROAD_PLANE': False}
cfg.DATA_CONFIG.AUTMENTATION.NOISE_PER_OBJECT = {"ENABLE":True,
                                                 "GT_LOC_NOISE_STD":[1.0, 1.0, 0.5],
                                                 "GT_ROT_UNIFORM_NOISE":[-0.78539816, 0.78539816]}
cfg.DATA_CONFIG.AUTMENTATION.NOISE_GLOGBAL_SCENE={"ENABLE":True,
                                                  "GLOBALE_ROT_UNIFORM_NOISE":[-0.78539816, 0.78539816],
                                                  "GLOBALE_SCALING_UNIFORM_NOISE":[0.95,1.05]}



cfg.MODEL = edict()
cfg.MODEL.VFE = edict()
cfg.MODEL.VFE = {"NAME":"MeanVoxelFeatureExtractor"}

cfg.MODEL.TRAIN=edict()
cfg.MODEL.TRAIN={"SPLIT":"train",
                 "EXTRA_TAG":"default",
                 "NMS_PRE_MAXSIZE":9000,
                 "NMS_POST_MAXSIZE":1024,
                 "RPN_NMS_THRESH":0.8,
                 "RPN_NMS_TYPE":"nms_gpu",
                 "OPTIMIZATION":{
                     "OPTIMIZER":"adam_onecycle",
                     "LR":0.003,
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
                     "GRAD_NORM_CLIP":10
                 }}

cfg.MODEL.TEST = edict()
cfg.MODEL.TEST ={
    "EXTRA_TAG":"default",
    "SPLIT":"val", #这里是修改test保存预测的目录
    "NMS_PRE_MAXSIZE":80,
    "NMS_POST_MAXSIZE":15,
    "RPN_NMS_THRESH":0.15,
    "RPN_NMS_TYPE":"nms_gpu",

    "MULITI_CLASSES_NMS":False,
    "NMS_THRESH":0.01,
    "NMS_TYPE":"nms_gpu",
    "SCORE_THRESH":0.1,
    "USE_RAW_SCORE":True,

    "NMS_PRE_MAXSIZE_LAST":4096,
    "NMS_POST_MAXSIZE_LAST":500,

    "RECALL_THRESH_LIST":[0.5,0.7],

    "EVAL_METRIC":"kitti",

    "BOX_FILTER":{
        "USE_IMAGE_AREA_FILTER":True,
        "LIMIT_RANGE":[0,-40,-3.0,70.4,40,3.0],
    }



}

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
                 "RPN_HEAD":{"NAME":"RPNV2",
                             "TARGET_CONFIG":{"ANCHOR_GENERATOR":[{"anchor_range":[0,-40,-1.78, 70.4,40,-1.78],
                                                                   "class_name":"Car",
                                                                   "matched_threshold":0.6,
                                                                   "rotation":[0,1.57],
                                                                   "sizes":[[1.6,3.9,1.56]],
                                                                   "unmatched_threshold":0.45
                                                                   },
                                                                  {"anchor_range":[0,-40,-0.6, 70.4,40,-0.6],
                                                                   "class_name":"Pedestrian",
                                                                   "matched_threshold":0.5,
                                                                   "rotation": [0, 1.57],
                                                                   "sizes": [[0.6, 0.8, 1.73]],
                                                                   "unmatched_threshold": 0.35
                                                                   },
                                                                  {"anchor_range":[0,-40,-0.6, 70.4,40,-0.6],
                                                                   "class_name":"Cyclist",
                                                                   "matched_threshold":0.5,
                                                                   "rotation": [0, 1.57],
                                                                   "sizes": [[0.6, 1.76, 1.73]],
                                                                   "unmatched_threshold": 0.35
                                                                   }
                                                                  ],
                                              "BOX_CODER":"ResidualCoder",
                                              "REGION_SIMILARITY_FN":"nearest_iou_similarity",
                                              "SAMPLE_POS_FRACTION":1,
                                              "SAMPLE_SIZE":512,
                                              "DOWNSAMPLED_FACTOR":8
                                              },
                             "ARGS":{"use_norm":True,
                                     "concat_input":False,
                                     "num_input_features":256,
                                     "layer_num":[5,5],
                                     "layer_strides":[1,2],
                                     "num_filters":[128,256],
                                     "upsample_strides":[1,2],
                                     "num_upsample_filter":[256,256],
                                     "encode_background_as_zeros":True,

                                     "use_direction_classifier":True,
                                     "num_direction_bins":2,
                                     "dir_offset":0.78539,
                                     "dir_limit_offset":0.0,
                                     "use_binary_dir_classifier":False}

                             }
                 }



cfg.MODEL.LOSSES=edict()
cfg.MODEL.LOSSES = {"RPN_REG_LOSS":"smooth-l1",
                    "RCNN_CLS_LOSS":"BinaryCrossEntropy",
                    "RCNN_REG_LOSS":"smooth-l1",
                    "CORNER_LOSS_REGULARIZION":True,
                    "LOSS_WEIGHT":{"rpn_cls_weight":1.0,
                                   "rpn_loc_weight":2.0,
                                   "rpn_dir_weight":0.2,

                                   "rcnn_cls_weight":1.0,
                                   "rcnn_reg_weight":1.0,
                                   "rcnn_corner_weight":1.0,
                                   "code_weights":[1.0,1.0,1.0,1.0,1.0,1.0,1.0]
                                   }
                    }
cfg.MODEL.RCNN = edict()
cfg.MODEL.RCNN= {
    "NAME":"SpconvRCNN", #FCRCNN,SpconvRCNN
    "ENABLE":True,
    "NUM_POINT_FEATURES":16,
    "ROI_AWARE_POOL_SIZE":12,
    "SHARED_FC":[128,256,256,256],
    "CLS_FC":[256,256],
    "REG_FC":[256,256],
    "DP_RATIO":0.3,

    "TARGET_CONFIG":{
        "BOX_CODER":"ResidualCoder",
        "ROI_PER_IMAGE":128,
        "FG_RATIO":0.5,
        "HARD_BG_RATIO":0.7,

        "CLS_SCORE_TYPE":"roi_iou",

        "CLS_FG_THRESH":0.75,
        "CLS_GB_THRESH":0.25,
        "CLS_GB_THRESH_L0":0.1,

        "REG_FG_THRESH":0.55
    },
}
DEBUG_DIR =os.path.join("/home/liang/PVRCNN-V0/","debug_file")

cfg.DEBUG =edict()
cfg.DEBUG={"ENABLE":True,
           "DEBUG_DIR":DEBUG_DIR,
           }

#这部分是train的args参数
cfg.train_args = edict()
cfg.train_args={"BATCH_SIZE" : 4,
                "pretrained_model":None,
                "LOCAL_RANK":0,
                "ckpt":"/home/liang/PVRCNN-V0/ckpt/",
                "ckpt_interval":1,
                "max_ckpt_save_num":50,
                "epoch":50}


#产生数据增强中的gt database的info路径和数据路径
cfg.GT_DATABASE_INFO="/home/liang/PVRCNN-V0/database_information"
cfg.GT_DATABASE = "/home/liang/PVRCNN-V0/database"



#关于test的参数也可以在这里修该
cfg.TEST_ARGS =edict()
cfg.TEST_ARGS = {
    "ckpt":"/home/liang/PVRCNN-V0/ckpt/",
    "EXTRA_TAG":"default",#for experiment
    "eval_all":False,#是否使用最终的模型进行eval
    "ckpt_id":"49",#预测哪个ckpt的模型
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