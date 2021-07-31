import numpy as np
from pvdet.tools.config import cfg
import torch.nn as nn
import torch
from pvdet.model.detectors.detector3d import Detector3d
import spconv
from pvdet.model.model_utils.proposal_layer import proposal_layer
import pickle
import os
from pvdet.model.RCNN import rcnn_modules
import time


class Part2net(Detector3d):
    def __init__(self,num_class,dataset):
        super().__init__(num_class, dataset)

        self.sparse_shape = dataset.voxel_generator.grid_size[::-1] + [1, 0, 0]
        self.print_info = cfg.print_info
        self.build_net(cfg.MODEL)


    def forward_rpn(self, voxels, num_points, coordinates, batch_size, **kwargs):
        # RPN inference
        with torch.set_grad_enabled((not cfg.MODEL.RPN.PARAMS_FIXED) and self.training):
            voxel_features = self.vfe(
                features=voxels,
                num_voxels=num_points,
                # coords=coordinates
            )

            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=coordinates,
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            start = time.time()
            unet_ret_dict = self.rpn_net(
                input_sp_tensor,
                **kwargs
            )
            if self.print_info:
                print("conv3d_net spend time ",time.time()-start)
            batch_dict = {"batch_size":batch_size,
                          "points":kwargs["points"],
                          "spatial_features":unet_ret_dict["spatial_features"],
                          "spatial_features_stride":unet_ret_dict["spatial_features_stride"],
                          "multi_scale_3d_features":unet_ret_dict["multi_scale_3d_features"],
                          "gt_boxes":kwargs.get("gt_boxes",None),
                          "sample_idx":kwargs["sample_idx"]}
            start = time.time()
            batch_dict = self.voxel_sa(batch_dict)
            if self.print_info:
                print("SA_module spend time:", (time.time() - start)/batch_size)
            start = time.time()
            rpn_preds_dict = self.rpn_head(
                unet_ret_dict['spatial_features'],
                **{'gt_boxes': kwargs.get('gt_boxes', None)}
            )
            if self.print_info:
                print("total conv2d spend time",(time.time()-start)/batch_size)
            rpn_preds_dict.update(unet_ret_dict)
            rpn_preds_dict.update(batch_dict)
            rpn_preds_dict = self.point_head(rpn_preds_dict)

        return rpn_preds_dict

    def forward_rcnn(self,rpn_preds_dict):
        start=time.time()
        rcnn_ret_dict = self.pv_rcnn(rpn_preds_dict)
        # print("RCNN spend time ", (time.time()-start)/rpn_preds_dict["batch_size"])
        return rcnn_ret_dict




    def forward(self, input_dict):

        start = time.time()
        rpn_ret_dict = self.forward_rpn(**input_dict)
        if self.print_info:
            print("rpn spend time ", (time.time()-start)/input_dict["batch_size"])

        rcnn_ret_dict = self.forward_rcnn( rpn_ret_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(rcnn_ret_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            start = time.time()
            preds_dict, recall_dict = self.post_processing(rcnn_ret_dict)
            print("post_processing spend time:", time.time() - start)
            return preds_dict, recall_dict

    def get_training_loss(self, rcnn_ret_dict):
        dis_play_dict ={}
        rpn_loss,tb_dict = self.rpn_head.get_loss()
        point_loss,tb_point_dict = self.point_head.get_loss()
        rcnn_loss,tb_rcnn_dict = self.pv_rcnn.get_loss()
        loss = rpn_loss+point_loss+rcnn_loss
        tb_dict.update(tb_point_dict)
        tb_dict.update(tb_rcnn_dict)
        dis_play_dict["rcnn_fg_reg_num"] = tb_dict['rcnn_fg_reg_num']
        dis_play_dict["loss"] = loss.item()
        if loss==None:
            print("错误")
            raise ValueError
        return loss,tb_dict,dis_play_dict











