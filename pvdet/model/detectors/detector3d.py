from pvdet.tools.config import cfg
from pvdet.model.VFE.vfe_layer import MeanVoxelFeatureExtractor
from pvdet.model.RPN.rpn_unet import UNetV2
import torch.nn as nn
import torch
from pvdet.model.bbox_head.rpn_head import RPNV2
from pvdet.model.RCNN.rcnn_net import SpconvRCNN
from pvdet.model.RCNN import rcnn_modules
import os
from pvdet.dataset.utils import common_utils
import numpy as np
from pvdet.ops.iou3d_nms import iou3d_nms_utils
from pvdet.dataset.utils import box_utils
from pvdet.model.VoxelSA.voxel_sa_utils import VoxelSetAbstraction
from pvdet.dataset.Dataset import build_data_loader
from pvdet.model.bbox_head.point_head_simple import PointHeadSimple
from pvdet.model.RCNN.pv_rcnn import PV_Rcnn



class Detector3d(nn.Module):
    def __init__(self,num_class,dataset):
        super().__init__()
        self.num_class = num_class
        self.dataset = dataset
        self.grid_size = dataset.voxel_generator.grid_size
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.vfe = self.rpn_net = self.rpn_head = self.rcnn_net= self.voxel_sa = self.point_head = None



    def build_net(self,model_cfg):
        model_info_dict = {
            "grid_size":self.dataset.grid_size,
            "voxel_size":self.dataset.voxel_size,
            "num_rawpoint_features": self.dataset.num_point_features,
            "point_cloud_range":self.dataset.point_cloud_range,
            "num_bev_features":cfg.MODEL.RPN.RPN_HEAD.num_bev_features
        }
        vfe_cfg = model_cfg.VFE
        self.vfe =MeanVoxelFeatureExtractor(
            num_input_features=cfg.DATA_CONFIG.NUM_POINT_FEATURES['use'],
            voxel_size=cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE,
            pc_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
            **vfe_cfg.ARGS
        )

        voxel_feature_num = self.vfe.get_output_feature_dim()

        rpn_cfg = model_cfg.RPN
        self.rpn_net = UNetV2(
            input_channels=voxel_feature_num,
            **rpn_cfg.BACKBONE.ARGS)
        rpn_head_cfg = model_cfg.RPN.RPN_HEAD
        self.rpn_head = RPNV2(
            num_class=self.num_class,
            args=rpn_head_cfg.ARGS,
            grid_size=self.grid_size,
            anchor_target_cfg=rpn_head_cfg.TARGET_CONFIG
                              )

        self.voxel_sa = VoxelSetAbstraction(
            model_cfg=cfg.MODEL.VOXEL_SA,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
            )
        model_info_dict.update({"num_point_features":self.voxel_sa.num_point_features,
                                    "num_point_features_before_fusion":self.voxel_sa.num_point_features_before_fusion})
        if cfg.MODEL.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']

        self.point_head = PointHeadSimple(
            model_cfg=cfg.MODEL.POINT_HEAD,
            input_channels=num_point_features,
            num_class=self.num_class if not cfg.MODEL.POINT_HEAD.CLASS_AGNOSTIC else 1,
        )
        self.pv_rcnn = PV_Rcnn(num_class=1,channel_in = model_info_dict["num_point_features"],model_cfg = cfg.MODEL.PV_RCNN)



    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            pass

    def load_params_from_file(self,filename,logger,to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        if logger is not None:
            logger.info("****Load paramters from checkpoint %s to %s" % (filename,"CPU" if to_cpu else "GPU"))
        loc_type = torch.device("cpu") if to_cpu else None
        checkpoint = torch.load(filename,map_location=loc_type)
        model_state_disk = checkpoint["model_state"]

        if "version" in checkpoint:
            logger.info("===>checkpoint trained from version:%s"%checkpoint["version"])

        update_model_state={}
        for key,val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape==model_state_disk[key].shape:
                update_model_state[key]= val
        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info("Not update weight %s: %s"%(key,str(state_dict[key].shape)))
        logger.info("==>Done (load %d/%d)"%(len(update_model_state),len(self.state_dict())))

    def load_params_with_optimizer(self,filename,to_cpu=False,optimizer=None,logger =None):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info("==>Loading paramters from checkpoint %s to %s"% (filename,"CPU" if to_cpu else "GPU"))
        loc_type =torch.device("cpu") if to_cpu else None
        checkpoint = torch.load(filename,map_location=loc_type)
        epoch = checkpoint.get("epoch",-1)
        it = checkpoint.get("it",0.0)

        self.load_state_dict(checkpoint["model_state"])

        if optimizer is not None:
            if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
                logger.info("==>Loading optimizer parameters from checkpoint %s to %s"
                            %(filename,"CPU" if to_cpu else "GPU"))
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            else:
                assert filename[-4] ==".",filename
                src_file,ext = filename[:-4],filename[:-3]
                optimizer_filename = "%s_optim.%s" %(src_file,ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it,epoch



    def post_processing(self, batch_dict):
        post_process_config = cfg.MODEL.POST_PROCESSING
        box_preds = batch_dict["rcnn_box_preds"]
        cls_preds = batch_dict["rcnn_cls_preds"]
        batch_size = batch_dict["batch_size"]

        pred_dicts = []
        recall_dict = {}
        for index in range(batch_size):
            cur_box_preds = box_preds[index]
            cur_cls_preds = cls_preds[index]
            cur_normal_cls_preds = torch.sigmoid(cur_cls_preds)
            label_preds = batch_dict["roi_labels"][index]
            selected,selected_score = self.class_final_nms(
                cur_normal_cls_preds,
                cur_box_preds,
                post_process_config.NMS_CONFIG,
                post_process_config.SCORE_THRESH)
            final_box = cur_box_preds[selected]
            final_score = selected_score
            final_cls = label_preds[selected]

            recall_dict = self.generate_recall_record(index,cur_box_preds,batch_dict,recall_dict,
                                        post_process_config.RECALL_THRESH_LIST)
            single_batch_dict = {
                "pred_boxes":final_box,
                "pred_labels":final_cls,
                "pred_scores":final_score
            }

            pred_dicts.append(single_batch_dict)
        return pred_dicts,recall_dict


    def class_final_nms(self,box_scores,box_preds,nms_config,score_thresh=None):
        """
        box_scores:(N,)
        box_preds:(N,7)
        score thresh:(1,)
        """
        box_scores_src = box_scores
        score_mask = (box_scores>score_thresh).squeeze(dim=-1)
        box_scores = box_scores[score_mask]
        box_preds = box_preds[score_mask]
        selected =[]
        if box_scores.shape[0]>0:
            score_for_nms,indices = torch.topk(
                box_scores.squeeze(dim=-1),k=min(nms_config.NMS_PRE_MAXSIZE,box_scores.shape[0]))
            box_for_nms = box_preds[indices]
            keep_id,_ = iou3d_nms_utils.nms_gpu(box_for_nms,score_for_nms,nms_config.NMS_THRESH)
            selected = indices[keep_id[:nms_config.NMS_POST_MAXSIZE]]

            origin_idx = torch.nonzero(score_mask).view(-1)
            selected = origin_idx[selected].view(-1)
        return selected,box_scores_src[selected].view(-1)


    def generate_recall_record(self,index, box_preds,batch_dict,recall_dict, thresh_list=(0.5, 0.7)):
        rois = batch_dict["rois"][index]
        if batch_dict["gt_boxes"] is not None:
            gt_boxes = batch_dict["gt_boxes"][index]
            cur_gt = gt_boxes
            k = gt_boxes.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_rois = rois
            if cur_gt.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds, cur_gt[:, :7])
                iou3d_rois = iou3d_nms_utils.boxes_iou3d_gpu(cur_rois, cur_gt[:, :7])
                if recall_dict.__len__() == 0:
                    recall_dict["gt"] = 0
                    for thresh in thresh_list:
                        recall_dict["roi_%s" % str(thresh)] = 0
                        recall_dict["rcnn_%s" % str(thresh)] = 0

                recall_dict["gt"] += cur_gt.shape[0]
                for thresh in thresh_list:
                    rois_iou_max = iou3d_rois.max(dim=-1)[0]
                    rcnn_iou_max = iou3d_rcnn.max(dim=-1)[0]
                    recall_dict["roi_%s" % str(thresh)] += (rois_iou_max > thresh).sum().item()
                    recall_dict["rcnn_%s" % str(thresh)] += (rcnn_iou_max > thresh).sum().item()
        else:
            return {}

        return recall_dict









if __name__ == '__main__':
    dataset,_,_ = build_data_loader()
    model_info_dict = {
        "grid_size": dataset.grid_size,
        "voxel_size": dataset.voxel_size,
        "num_rawpoint_features": dataset.num_point_features,
        "point_cloud_range": dataset.point_cloud_range,
        "num_bev_features": cfg.MODEL.RPN.RPN_HEAD.num_bev_features
    }
    voxel_sa = VoxelSetAbstraction(
        model_cfg=cfg.MODEL.VOXEL_SA ,
        voxel_size=model_info_dict['voxel_size'],
        point_cloud_range=model_info_dict['point_cloud_range'],
        num_bev_features=model_info_dict['num_bev_features'],
        num_rawpoint_features=model_info_dict['num_rawpoint_features']
    )
    print("done")