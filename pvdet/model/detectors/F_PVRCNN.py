import torch
import spconv
from new_train.model.detector3d.detector_utils import Detector
import time
from new_train.config import cfg


class FPVdet(Detector):
    def __init__(self,dataset,logger):
        super().__init__(dataset,logger)


    def rpn_forward(self,batch_data
                    ):
        start = time.time()
        voxels = batch_data["voxels"]
        num_points = batch_data["num_points"]
        coordinates = batch_data["coordinates"]
        batch_size = batch_data["batch_size"]
        if cfg.print_info:
            print("copy spend time: ",time.time()-start)
        # voxel_centers = batch_data["voxel_centers"]
        # gt_boxes = batch_data.get("gt_boxes",None)
        # sample_idx = batch_data["sample_idx"]
        with torch.set_grad_enabled(self.training):
            start = time.time()
            voxels_mean = self.vfe(voxels,num_points)
            spconv_tensor = spconv.SparseConvTensor(
                features=voxels_mean,
                indices=coordinates,
                spatial_shape=self.spatial_shapes,
                batch_size=batch_size)
            if cfg.print_info:
                print("vfe and voxelization ",time.time()-start)
            start = time.time()
            conv3d_out = self.conv_3d(spconv_tensor,
                                      batch_data)
            if cfg.print_info:
                print("total conv3d spend time:",(time.time() - start))

            start = time.time()
            batch_data.update(conv3d_out)
            batch_data = self.SA_module(batch_data)
            if cfg.print_info:
                print("total SA_module spend time:", (time.time() - start))

            start = time.time()
            conv2d_out = self.conv_2d(
                batch_data["spatial_features"],
                batch_data.get("gt_boxes",None))
            if cfg.print_info:
                print("Conv2d spend time:", (time.time() - start)/batch_size)
            start = time.time()
            batch_data.update(conv2d_out)
            batch_data = self.point_head(batch_data)
            if cfg.print_info:
                print("total point_head spend time:", (time.time() - start)/batch_size)

            return batch_data
    def forward(self,batch_data):
        start = time.time()
        batch_data = self.rpn_forward(
            batch_data,
        )
        if cfg.print_info:
            print("total rpn spend time:", (time.time() - start) )
        start = time.time()
        rcnn_ret_dict = self.rcnn(batch_data)
        if cfg.print_info:
            print("total rcnn spend time:", (time.time() - start))

        start = time.time()
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            if cfg.print_info:
                print("total get_loss spend time:", (time.time() - start) / batch_data["batch_size"])
            return ret_dict, tb_dict, disp_dict
        else:
            start = time.time()
            preds_dict, recall_dict = self.post_processing(rcnn_ret_dict)
            print("post_processing spend time:",time.time()-start)
            return preds_dict, recall_dict



    def get_training_loss(self,):
        dis_play_dict ={}
        seg_loss = self.conv_3d.get_loss()
        rpn_loss,tb_dict = self.conv_2d.get_loss()
        point_loss,tb_point_dict = self.point_head.get_loss()
        rcnn_loss,tb_rcnn_dict = self.rcnn.get_loss()
        loss = rpn_loss+point_loss+rcnn_loss+seg_loss
        tb_dict.update(tb_point_dict)
        tb_dict.update(tb_rcnn_dict)
        dis_play_dict["rcnn_fg_reg_num"] = tb_dict['rcnn_fg_reg_num']
        dis_play_dict["loss"] = loss.item()
        if loss==None:
            print("错误")
            raise ValueError
        return loss,tb_dict,dis_play_dict

