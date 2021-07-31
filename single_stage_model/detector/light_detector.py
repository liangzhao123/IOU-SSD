from single_stage_model.detector.single_model_baseline import SSDbase

import torch
import spconv
from single_stage_model.configs.single_stage_config import cfg
import time
from single_stage_model.detector.single_model_baseline import SSDbase


class LightDetector(SSDbase):
    def __init__(self,logger,config,cfg):
        super().__init__(logger,config,cfg)


    def rpn_forward(self,batch_data
                    ):
        start = time.time()
        voxels = batch_data["voxels"]
        num_points = batch_data["num_points"]
        coordinates = batch_data["coordinates"]
        batch_size = batch_data["batch_size"]

        with torch.set_grad_enabled(self.training):
            start = time.time()
            voxels_mean = self.vfe(voxels,num_points)
            batch_data["voxels_mean"] = voxels_mean
            spconv_tensor = spconv.SparseConvTensor(
                features=voxels_mean,
                indices=coordinates,
                spatial_shape=self.spatial_shapes,
                batch_size=batch_size)

            batch_data = self.conv_3d(spconv_tensor,
                                      batch_data)

            start = time.time()
            batch_data = self.conv_2d(batch_data)
            # print("conv2d spend time:",(time.time()-start)/batch_size)
            batch_data = self.detect_head(batch_data)
            if self.using_iou_head:
                batch_data = self.iou_head(batch_data)
            return batch_data
    def forward(self,batch_data):
        start = time.time()
        batch_data = self.rpn_forward(
            batch_data,
        )

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            # print(tb_dict)
            ret_dict = {
                'loss': loss
            }

            return ret_dict, tb_dict, disp_dict
        else:
            start = time.time()
            preds_dict, recall_dict = self.post_processing_for_single_stage_model(batch_data,save_proposals=self.cfg.model.post_processing.get("save_proposals",False))
            print("post process spend time:", (time.time() - start)/batch_data["batch_size"])
            return preds_dict, recall_dict



    def get_training_loss(self,):
        dis_play_dict = {}
        rpn_loss,tb_dict = self.detect_head.get_loss()
        if self.using_iou_head:
            iou_loss,tb_dict_ = self.iou_head.get_loss()
            loss = rpn_loss + iou_loss
            tb_dict.update(tb_dict_)
        else:
            loss = rpn_loss

        dis_play_dict["loss"] = loss.item()
        if loss==None:
            print("错误")
            raise ValueError
        return loss,tb_dict,dis_play_dict