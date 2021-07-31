
import pvdet.model.pointnet2.pointnet2_stack.pointnet2_modules as pointnet2_stack_modules
import pvdet.model.pointnet2.pointnet2_stack.pointnet2_utils as pointnet2_stack_utils
import torch
import torch.nn as nn
import numpy as np
from pvdet.dataset.utils import common_utils
import time
from vis_points.fps_utils import fps_func_utils
from new_train.config import cfg
from fps_ops.utils import fps_f,fps_f_v1,fps_f_v2
import pickle
# acjasbcjbachjs
def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class VoxelSA_old(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        # self.print_info = cfg.print_info
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = np.array(point_cloud_range)

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                # print(src_name)
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
            mlps = SA_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [mlps[k][0]] + mlps[k]
            cur_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg[src_name].POOL_RADIUS,
                nsamples=SA_cfg[src_name].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool',
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += sum([x[-1] for x in mlps])

        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            mlps = SA_cfg['raw_points'].MLPS
            for k in range(len(mlps)):
                mlps[k] = [num_rawpoint_features - 3] + mlps[k]

            self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg['raw_points'].POOL_RADIUS,
                nsamples=SA_cfg['raw_points'].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool'
            )
            c_in += sum([x[-1] for x in mlps])

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.points_out_channels = self.model_cfg.NUM_OUTPUT_FEATURES
        self.points_out_channels_befor_fusion = c_in

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):

        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def get_sampled_points(self, batch_dict,using_raw_points=False,using_fusion=False):
        start = time.time()
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_voxel_centers = batch_dict["voxel_centers"]
            batch_indices_voxel = batch_dict['coordinates'][:, 0].long()
            if using_raw_points:
                src_points = batch_dict['points'][:, 1:4]
                batch_indices = batch_dict['points'][:, 0].long()
        else:
            raise NotImplementedError
        if cfg.print_info:
            print("copy and intital in fps spend time ",(time.time()-start)/batch_dict["batch_size"])
        keypoints_list = []
        seg_score = batch_dict["seg_score"]
        batch_cnt = torch.zeros(batch_size,dtype=torch.int64).cuda()
        for i in range(batch_size):
            batch_cnt[i] = (batch_dict["coordinates"][:,0]==i).sum()

        for bs_idx in range(batch_size):
            start = time.time()
            bs_mask_voxel = (batch_indices_voxel == bs_idx)
            if using_raw_points:
                bs_mask_points = (batch_indices == bs_idx)
                sampled_points = src_points[bs_mask_points].unsqueeze(dim=0)
            sampled_voxels = src_voxel_centers[bs_mask_voxel].unsqueeze(dim=0)  # (1, N, 3)
            cur_seg_score_pred = seg_score[bs_mask_voxel].view(-1,2)


            # if bs_idx==0:
            #     sampled_points = src_points[:batch_cnt[bs_idx]].unsqueeze(dim=0)
            #     cur_seg_score_pred = seg_score[:batch_cnt[bs_idx]].view(-1,2)
            # elif bs_idx==batch_size-1:
            #     sampled_points = src_points[-batch_cnt[bs_idx]:].unsqueeze(dim=0)
            #     cur_seg_score_pred = seg_score[-batch_cnt[bs_idx]:].view(-1,2)
            # elif bs_idx<(batch_size-1) and bs_idx>0:
            #     sampled_points = src_points[batch_cnt[bs_idx-1]:(batch_cnt[bs_idx-1]+batch_cnt[bs_idx])].unsqueeze(dim=0)
            #     cur_seg_score_pred = seg_score[batch_cnt[bs_idx-1]:(batch_cnt[bs_idx-1]+batch_cnt[bs_idx])].view(-1,2)

            if cfg.print_info:
                print("prepare for loop in fps spend time %d: "%bs_idx, time.time() - start)

            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                start_time = time.time()
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()
                FPS_time =time.time() -start_time

                # print("FPS %d -> %d %.6f" % (sampled_points.shape[1],self.model_cfg.NUM_KEYPOINTS,FPS_time))#计算FPS需要的时间

                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            elif self.model_cfg.SAMPLE_METHOD == 'SegFPS_v1':
                keypoints = fps_f_v1(sampled_points,
                      seg_score=cur_seg_score_pred,
                      npoints=cfg.MODEL.SA_module["num_keypoints"])
                # keypoints_list.append(fps_func_utils.fps_with_featres(
                #     sampled_points,cur_fg_point_idx,cfg.MODEL.SA_module["num_keypoints"])
                # )
            elif self.model_cfg.SAMPLE_METHOD == "SegFPS_v2":
                keypoints_from_SegFPS = fps_f_v2(sampled_voxels,
                                     seg_score=cur_seg_score_pred,
                                     npoints=cfg.MODEL.SA_module["num_keypoints"])
                if using_raw_points:
                    cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                    ).long()
                    keypoints_from_fps = sampled_points[0][cur_pt_idxs[0]]
                else:
                    cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                        sampled_voxels[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                    ).long()
                    keypoints_from_fps = sampled_voxels[0][cur_pt_idxs[0]]
                keypoints_before_fusion = torch.cat((keypoints_from_SegFPS, keypoints_from_fps), dim=0).squeeze(dim=0)
                if using_fusion:
                    raise NotImplementedError
                    npoints = self.model_cfg.NUM_KEYPOINTS/2
                    cur_pt_idxs_fuse = pointnet2_stack_utils.furthest_point_sample(
                        keypoints_before_fusion[:, :, 0:3].contiguous(), npoints
                    ).long()
                    keypoints = keypoints_before_fusion[0][cur_pt_idxs_fuse[0]]
                else:
                    keypoints = keypoints_before_fusion

                if bs_idx>=10:
                    vis_dict = {"voxel_centers":sampled_voxels.cpu().numpy(),
                                "keypoints":keypoints.cpu().numpy(),
                                "keypoints_from_fps":keypoints_from_fps.cpu().numpy(),
                                "keypoints_from_SegFPS":keypoints_from_SegFPS.cpu().numpy(),
                                "gt_boxes":batch_dict["gt_boxes"][bs_idx].cpu().numpy(),
                                "sample_idx":batch_dict["sample_idx"][bs_idx]}
                    import os
                    filename = "vis_keypoints_%s.pth" % vis_dict["sample_idx"]
                    path = os.path.join(cfg.CODE_DIR,"preds",filename)
                    with open(path,"wb") as f:
                        pickle.dump(vis_dict,f)
            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints.view(batch_size,-1,3)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        start = time.time()
        keypoints = self.get_sampled_points(batch_dict)
        if cfg.print_info:
            print("total FPS spend time:", (time.time() - start) / batch_dict["batch_size"])
        start = time.time()
        point_features_list = []
        start = time.time()
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
                bev_stride=cfg.MODEL.SA_module["bev_stride"]
            )
            point_features_list.append(point_bev_features)

        if cfg.print_info:
            print("bev spend time:",(time.time()-start)/batch_dict["batch_size"])

        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)


        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            raw_points = batch_dict['points']
            xyz = raw_points[:, 1:4]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()
            point_features = raw_points[:, 4:].contiguous() if len(raw_points) > 4 else None
            start = time.time()
            pooled_points, pooled_features = self.SA_rawpoints(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features,
            )
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))
        if cfg.print_info:
            print("rawpoint spend time:",(time.time()-start)/batch_dict["batch_size"])

        start = time.time()
        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict[src_name].indices
            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            pooled_points, pooled_features = self.SA_layers[k](
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=batch_dict[src_name].features.contiguous(),
            )
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))
        if cfg.print_info:
            print("mutil_sclan_feat spend time:",(time.time()-start)/batch_dict["batch_size"])

        point_features = torch.cat(point_features_list, dim=2)

        batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)

        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

        batch_dict['point_features'] = point_features  # (BxN, C)
        batch_dict['point_coords'] = point_coords  # (BxN, 4)

        return batch_dict

if __name__ == '__main__':
    a = torch.tensor([1,2,3,4,5,6,7]).cuda()
    b = a[-3:]
    print("A")
# FPS 0.000129
# grouper time 0.022824
# FP time 0.001062
# grouper time 0.004589
# FP time 0.000585
# grouper time 0.003387
# FP time 0.000606
# grouper time 0.003742
# FP time 0.000589
# grouper time 0.004945
# FP time 0.000585
# grouper time 0.005700
# FP time 0.000585
# grouper time 0.003737
# FP time 0.000952
# grouper time 0.004677
# FP time 0.001265
# grouper time 0.002503
# FP time 0.001215
# grouper time 0.003641
# FP time 0.001436


# FPS 16900 -> 2048 0.000079
# FPS 20695 -> 2048 0.000062
# grouper time 37595->4096 0.023145
# Sequential(
#   (0): Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000668
# grouper time 37595->4096 0.004496
# Sequential(
#   (0): Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000370
# grouper time 28596->4096 0.002947
# Sequential(
#   (0): Conv2d(19, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000369
# grouper time 28596->4096 0.003745
# Sequential(
#   (0): Conv2d(19, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000360
# grouper time 49856->4096 0.004737
# Sequential(
#   (0): Conv2d(35, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000335
# grouper time 49856->4096 0.006144
# Sequential(
#   (0): Conv2d(35, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000275
# grouper time 34655->4096 0.003503
# Sequential(
#   (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000645
# grouper time 34655->4096 0.004945
# Sequential(
#   (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000781
# grouper time 16409->4096 0.001966
# Sequential(
#   (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000650
# grouper time 16409->4096 0.003222
# Sequential(
#   (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000773

#########################################     推理过程 44    ########################################################################

# grouper time 18839->2048 0.016024
# Sequential(
#   (0): Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000233
# grouper time 18839->2048 0.002704
# Sequential(
#   (0): Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000188
# len(groupers) 2
# grouper time 15325->2048 0.002134
# Sequential(
#   (0): Conv2d(19, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000191
# grouper time 15325->2048 0.002425
# Sequential(
#   (0): Conv2d(19, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000188
# len(groupers) 2
# grouper time 29571->2048 0.003617
# Sequential(
#   (0): Conv2d(35, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000191
# grouper time 29571->2048 0.004259
# Sequential(
#   (0): Conv2d(35, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000189
# len(groupers) 2
# grouper time 20886->2048 0.002710
# Sequential(
#   (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000190
# grouper time 20886->2048 0.003230
# Sequential(
#   (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000188
# len(groupers) 2
# grouper time 9728->2048 0.001536
# Sequential(
#   (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000190
# grouper time 9728->2048 0.001809
# Sequential(
#   (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000188
# len(groupers) 2
# grouper time 2048->21600 0.001238
# Sequential(
#   (0): Conv2d(131, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000190
# grouper time 2048->21600 0.006473
# Sequential(
#   (0): Conv2d(131, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000188
# 第 44 帧用的时间: 0.09359502792358398


###########################################推理过程 44 （nsample从2048改为512） ####################################################

# grouper time 18839->512 0.005101
# Sequential(
#   (0): Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000232
# grouper time 18839->512 0.002465
# Sequential(
#   (0): Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000195
# len(groupers) 2
# grouper time 15325->512 0.002117
# Sequential(
#   (0): Conv2d(19, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000195
# grouper time 15325->512 0.002131
# Sequential(
#   (0): Conv2d(19, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000192
# len(groupers) 2
# grouper time 29571->512 0.004604
# Sequential(
#   (0): Conv2d(35, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000195
# grouper time 29571->512 0.004260
# Sequential(
#   (0): Conv2d(35, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000196
# len(groupers) 2
# grouper time 20886->512 0.002710
# Sequential(
#   (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000195
# grouper time 20886->512 0.003162
# Sequential(
#   (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000192
# len(groupers) 2
# grouper time 9728->512 0.001537
# Sequential(
#   (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000193
# grouper time 9728->512 0.001717
# Sequential(
#   (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000196
# len(groupers) 2
# grouper time 512->21600 0.001070
# Sequential(
#   (0): Conv2d(131, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000195
# grouper time 512->21600 0.006304
# Sequential(
#   (0): Conv2d(131, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000193
# 第 44 帧用的时间: 0.08050274848937988



#################################################第44次迭代 将所有的SA层全部变成一次group

# grouper time 18839->2048 0.015263
# Sequential(
#   (0): Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000231
# len(groupers) 1
# grouper time 15325->2048 0.002129
# Sequential(
#   (0): Conv2d(19, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000197
# len(groupers) 1
# grouper time 29571->2048 0.003630
# Sequential(
#   (0): Conv2d(35, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000196
# len(groupers) 1
# grouper time 20886->2048 0.003534
# Sequential(
#   (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000199
# len(groupers) 1
# grouper time 9728->2048 0.001551
# Sequential(
#   (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000202
# len(groupers) 2
# grouper time 2048->21600 0.001260
# Sequential(
#   (0): Conv2d(131, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000198
# grouper time 2048->21600 0.006459
# Sequential(
#   (0): Conv2d(131, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
# ) ,FP time 0.000196
# 第 44 帧用的时间: 0.07588458061218262