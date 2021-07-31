from new_train.config import cfg
from pvdet.dataset.utils import box_utils
import time
import torch
import torch.nn as nn
import spconv
from functools import partial
from pvdet.dataset.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu,points_in_boxes_gpu
import numpy as np
from pvdet.tools.utils import loss_utils
from new_train.model.segnet.seg_net_utils import SegNet


class VEF(nn.Module):
    def __init__(self,num_channel):
        super().__init__()
        self.used_feature = num_channel
    def forward(self,voxels,num_per_voxel):
        mean_voxel = voxels.sum(dim=1)/ num_per_voxel.type_as(voxels).view(-1,1)
        return mean_voxel.contiguous()

class Conv_3d_net(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        # self.print_info = cfg.print_info
        norm_fn = partial(nn.BatchNorm1d,eps=1e-3, momentum=0.01)
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )

        block = self.post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0 if cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE[-1] in [0.1, 0.2] else (1, 0, 0)

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.seg_net = SegNet(16)
        self.seg_loss_func =loss_utils.SigmoidFocalClassificationLoss_v1(alpha=0.25, gamma=2.0)
        self.num_point_features = 128
        self.ret = {}

    def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0,
                       conv_type='subm', norm_fn=None):
        if conv_type == 'subm':
            m = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(
                spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    bias=False, indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(
                spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size,
                                           indice_key=indice_key, bias=False),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError
        return m


    def get_seg_target_v1(self,voxel_center,coordinates,gt_boxes_lidar,set_ignore_flag=True):
        batch_size = gt_boxes_lidar.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes_lidar.view(-1, gt_boxes_lidar.shape[-1]),
            extra_width=cfg.MODEL.Seg_net["gt_extend_width"],
        ).view(batch_size, -1, gt_boxes_lidar.shape[-1])

        n = voxel_center.shape[0]#voxel per frame
        voxel_cls_labels = voxel_center.new_zeros(voxel_center.shape[0]).long()
        one_hot_seg_target = torch.zeros((n, 2), dtype=torch.float32).cuda()
        
        for bs in range(self.batch_size):
            bs_indx = (coordinates[:,0]==bs)
            voxel_cls_labels_single  = voxel_cls_labels.new_zeros(sum(bs_indx))
            cur_voxels = (voxel_center[bs_indx])
            cur_gt_boxes = torch.tensor(gt_boxes_lidar[bs],dtype=torch.float32).cuda()
            seg_label = points_in_boxes_gpu(
                cur_voxels.unsqueeze(dim=0),
                cur_gt_boxes[:,:7].unsqueeze(dim=0).contiguous()
            ).long().squeeze(dim=0)
            fg_flag = (seg_label>=0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = points_in_boxes_gpu(
                    cur_voxels.unsqueeze(dim=0),
                    extend_gt_boxes[bs:bs + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts>=0)
                voxel_cls_labels_single[ignore_flag] = -1
            
            voxel_cls_labels_single[fg_flag] = 1
            voxel_cls_labels[bs_indx] = voxel_cls_labels_single

        cls_label_for_one_hot = (voxel_cls_labels * (voxel_cls_labels >= 0).long()).unsqueeze(dim=-1).long()
        one_hot_seg_target = one_hot_seg_target.scatter_(
            -1,cls_label_for_one_hot,1.0)
        self.ret["seg_label_src"] = voxel_cls_labels
        return one_hot_seg_target

    def get_seg_target_v0(self, voxel_center,
                          coordinates,
                          gt_boxes_lidar,
                          batch_size,
                          set_ignore_flag=False):
        batch_size = gt_boxes_lidar.shape[0]
        n = voxel_center.shape[0]
        one_hot_seg_target = torch.zeros((n, 2), dtype=torch.float32).cuda()
        voxel_cls_labels = voxel_center.new_zeros(voxel_center.shape[0]).long()
        for bs_idx in range(batch_size):
            cur_bs_idx = (bs_idx==coordinates[:,0])
            cur_voxel_centers = voxel_center[cur_bs_idx]
            cur_gt_box = gt_boxes_lidar[bs_idx]
            indx = points_in_boxes_cpu(cur_voxel_centers.cpu(),cur_gt_box[:,:7].cpu()).cuda()
            indx = indx.sum(axis=0)
            indx = torch.where(indx>0,torch.tensor(1).cuda(),torch.tensor(0).cuda())
            one_hot_seg_target[cur_bs_idx] = one_hot_seg_target[cur_bs_idx].scatter_(-1,indx.unsqueeze(dim=-1),1.0)
            voxel_cls_labels[cur_bs_idx] = indx
        self.ret["seg_label_src"] = voxel_cls_labels
        return one_hot_seg_target

    def get_loss(self,):
        batch_size = self.ret["batch_size"]
        seg_label_src = self.ret["seg_label_src"]
        postives = seg_label_src>0
        negitive = seg_label_src==0
        cls_target = self.ret["seg_target"]
        cls_pred = self.ret["seg_pred"]
        cls_weight = postives.float()*1.0+negitive.float()*1.0

        postives_normal = postives.sum(dim=0).float()
        cls_weight /= torch.clamp(postives_normal,min=1)
        cls_target = cls_target[...,1:]
        seg_loss_src = self.seg_loss_func(cls_pred,cls_target,cls_weight)
        seg_loss = seg_loss_src.sum()/batch_size
        seg_loss = seg_loss*cfg.MODEL.CONV3D["seg_loss_weight"]
        return seg_loss



    def forward(self,input_sp_tensor,
                batch_data):
        """
                :param voxel_features:  (N, C)
                :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
                :param batch_size:
                :return:
                """
        start = time.time()
        voxel_centers = batch_data["voxel_centers"]
        coordinates = batch_data["coordinates"]
        gt_boxes =  batch_data.get("gt_boxes",None)
        x = self.conv_input(input_sp_tensor)#[41,1600,1408]

        x_conv1 = self.conv1(x) #[41,1600,1408]
        x_conv2 = self.conv2(x_conv1)#[21,800,704]
        x_conv3 = self.conv3(x_conv2)#[11,400,352]
        x_conv4 = self.conv4(x_conv3)#[5,200,176]

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)#[2,200,176]
        spatial_features = out.dense()
        N, C, D, H, W = spatial_features.shape# batch size,128,2,200,176
        spatial_features = spatial_features.view(N,C*D,H,W) #[batch_size,128*2,200,176]
        start = time.time()
        seg_pred = self.seg_net(x_conv1)
        # print("seg_net spend time:",time.time()-start)
        point_cls_scores = torch.sigmoid(seg_pred)
        if cfg.print_info:
            print("sparse_conv3d spend time ",(time.time()-start)/batch_data["batch_size"])
        start = time.time()
        self.ret.update({"spatial_features": spatial_features,
                         "conv3d_1": x_conv1,
                         "conv3d_2": x_conv2,
                         "conv3d_3": x_conv3,
                         "conv3d_4": x_conv4,
                         "seg_pred": seg_pred,
                         "seg_score": point_cls_scores,
                         "batch_size": N})

        if self.training:
            self.batch_size = N
            strategy = "get_seg_target_v0"
            if strategy=="get_seg_target_v1":
                seg_target = self.get_seg_target_v1(voxel_centers,
                                             coordinates,
                                             gt_boxes)#(batch_size*n,n)
            elif strategy == "get_seg_target_v0":
                seg_target = self.get_seg_target_v0(voxel_centers,
                                                    coordinates,
                                                    gt_boxes,
                                                    batch_size=N)
            self.ret["seg_target"] = seg_target
            # loss = self.get_loss()
        if cfg.print_info:
            print("get_seg_target spend time ",(time.time()-start))
        return self.ret