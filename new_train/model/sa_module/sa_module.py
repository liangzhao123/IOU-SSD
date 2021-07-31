from fps_ops.utils import fps_f
from new_train.utils import common_utils
import time
import pvdet.model.pointnet2.pointnet2_stack.pointnet2_modules as pointnet2
import torch
import torch.nn as nn
from new_train.config import cfg
import numpy as np



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

class VoxelSA(nn.Module):
    def __init__(self,num_voxel_center_features=3,num_bev_features=256):
        super().__init__()
        self.features_src = cfg.MODEL.SA_module["feature_source"]
        self.conv3d_names = cfg.MODEL.SA_module["conv3d_names"]
        self.SA_layer = nn.ModuleList()
        self.conv3d_strides = {}
        self.point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        self.voxel_size = cfg.DATA_CONFIG.VOXEL_GENERATOR["VOXEL_SIZE"]
        self.using_voxel_center = cfg.MODEL.SA_module["using_voxel_center"]
        channel_in = 0
        if "voxel_centers" in self.features_src:
            mlps = cfg.MODEL.SA_module.sa_layer["voxel_centers"]["mlps"]
            for i in range(len(mlps)):
                if self.using_voxel_center:
                    mlps[i] = [num_voxel_center_features-3] + mlps[i]
                else:
                    mlps[i] = [num_voxel_center_features -3 +1] + mlps[i]
            nsamples = cfg.MODEL.SA_module.sa_layer.voxel_centers["nsamples"]
            pool_radius = cfg.MODEL.SA_module.sa_layer.voxel_centers["pool_radius"]
            self.SA_voxel_centers = pointnet2.StackSAModuleMSG(
                nsamples=nsamples,
                radii = pool_radius,
                mlps= mlps,
                use_xyz=True,
                pool_method = "max_pool"
            )
            channel_in += sum([x[-1]for x in mlps])
        if "conv_3d" in self.features_src:
            for feature_name in self.conv3d_names:
                if (feature_name in ["voxel_centers","bev"]):
                    continue
                mlps = cfg.MODEL.SA_module.sa_layer[feature_name]["mlps"]
                self.conv3d_strides[feature_name] = cfg.MODEL.SA_module.sa_layer[feature_name]["down_sample_fraction"]
                for k in range(len(mlps)):
                    mlps[k] = [mlps[k][0]] + mlps[k]
                nsamples = cfg.MODEL.SA_module.sa_layer[feature_name]["nsamples"]
                pool_radius = cfg.MODEL.SA_module.sa_layer[feature_name]["pool_radius"]
                cur_sa_layer = pointnet2.StackSAModuleMSG(
                    nsamples=nsamples,
                    radii = pool_radius,
                    mlps= mlps,
                    use_xyz=True,
                    pool_method = "max_pool"
                )
                self.SA_layer.append(cur_sa_layer)
                channel_in += sum([x[-1]for x in mlps])
        if "bev" in self.features_src:
            channel_in += num_bev_features
        feature_src_out_channel = cfg.MODEL.SA_module["point_features_out_channel"]
        self.features_src_fusion = nn.Sequential(
            nn.Linear(channel_in,feature_src_out_channel,bias=False),
            nn.BatchNorm1d(feature_src_out_channel),
            nn.ReLU()
        )
        self.points_out_channels = feature_src_out_channel
        self.points_out_channels_befor_fusion = channel_in
        self.ret = {}


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


    def get_fps_points(self,batch_data):
        # points = voxel_centers
        # seg_pred = seg_pred
        # batch_size = batch_size
        # coordinates = coordinates
        voxel_centers = batch_data["voxel_centers"]
        start = time.time()
        seg_score_pred,seg_label_pred = torch.max(batch_data["seg_pred"],dim=1)
        keypoints_list = []
        batch_indices = batch_data["coordinates"][:,0].long()
        for batch_idx in range(batch_data["batch_size"]):
            cur_bs_idx = (batch_indices==batch_idx)

            cur_seg_label_pred = seg_label_pred[cur_bs_idx]
            cur_points = voxel_centers[cur_bs_idx].unsqueeze(dim=0)

            cur_keypoints = fps_f(cur_points,cur_seg_label_pred,cfg.MODEL.SA_module["num_keypoints"])
            keypoints_list.append(cur_keypoints.unsqueeze(dim=0))


        keypoints = torch.cat(keypoints_list, dim=0)
        # print("total fps spend time:", (time.time() - start)/batch_data["batch_size"])
        return keypoints


    def forward(self,batch_data):
        start = time.time()
        keypoints = self.get_fps_points(batch_data)
        # print("F-FPS spend time:",(time.time()-start)/batch_data["batch_size"])
        start = time.time()
        point_wise_featurs_list = []
        start = time.time()
        if "bev" in self.features_src:
            bev_point_featurs = self.interpolate_from_bev_features(
                keypoints=keypoints,
                bev_features=batch_data["spatial_features"],
                batch_size=batch_data["batch_size"],
                bev_stride=cfg.MODEL.SA_module["bev_stride"],
            )
            point_wise_featurs_list.append(bev_point_featurs)
        # print("bev spend time:",(time.time()-start)/batch_data["batch_size"])
        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cout = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)

        start = time.time()
        if "voxel_centers" in self.features_src:
            if  self.using_voxel_center == True:
                xyz = batch_data["voxel_centers"]
                coordinate = batch_data["coordinates"]
                xyz_batch_cout = xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    xyz_batch_cout[bs_idx] = (coordinate[:,0] == bs_idx).sum()
                point_features = xyz[:,4:].contiguous() if (xyz.shape[1])>4 else None
            else:
                raw_points = batch_data['points']
                xyz = raw_points[:, 1:4]
                xyz_batch_cout = xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    xyz_batch_cout[bs_idx] = (raw_points[:, 0] == bs_idx).sum()
                point_features = raw_points[:, 4:].contiguous() if len(raw_points) > 4 else None
            pool_points, pool_features = self.SA_voxel_centers(
                xyz = xyz.contiguous(),
                xyz_batch_cnt = xyz_batch_cout,
                new_xyz = new_xyz,
                new_xyz_batch_cnt = new_xyz_batch_cout,
                features = point_features,
            )
            point_wise_featurs_list.append(pool_features.view(batch_size,num_keypoints,-1))
        # print("voxel_centers spend time:", (time.time() - start) / batch_data["batch_size"])

        start = time.time()

        for k, src_name in enumerate(self.conv3d_names):
            cur_coords = batch_data[src_name].indices
            xyz = common_utils.get_voxel_centers(
                cur_coords[:,1:4],
                downsample_times= self.conv3d_strides[src_name],
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            xyz_batch_cout = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cout[bs_idx] = (cur_coords[:,0]==bs_idx).sum()
            pool_points,pool_features = self.SA_layer[k](
                xyz = xyz.contiguous(),
                xyz_batch_cnt = xyz_batch_cout,
                new_xyz = new_xyz,
                new_xyz_batch_cnt = new_xyz_batch_cout,
                features = batch_data[src_name].features.contiguous()
            )
            point_wise_featurs_list.append(pool_features.view(batch_size,num_keypoints,-1))
        # print("mutil_sclan_feat spend time:",(time.time()-start)/batch_data["batch_size"])
        point_features = torch.cat(point_wise_featurs_list,dim=2)
        batch_idx = torch.arange(batch_size,device=keypoints.device).view(-1,1).repeat(1,keypoints.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1,1).float(),keypoints.view(-1,3)),dim=1)
        self.ret["point_features_before_fusion"] =  point_features.view(-1,point_features.shape[-1])
        point_features = self.features_src_fusion(point_features.view(-1,point_features.shape[-1]))

        self.ret["point_features"] = point_features
        self.ret["point_coords"] = point_coords
        batch_data.update(self.ret)
        return batch_data