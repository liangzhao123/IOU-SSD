import os
import torch
from tensorboardX import SummaryWriter
import time
from pvdet.tools.config import cfg
import re
import datetime
from pvdet.dataset.utils import common_utils
from pvdet.tools.config import log_cfg_to_file
import torch.distributed  as dist
from pvdet.dataset.Dataset import build_data_loader
from pvdet.model.detectors import build_netword
import glob
from pvdet.tools.eval_utils import eval_utils
from easydict import EasyDict as edict
from pvdet.tools.train_utils.train_utils import example_convert_to_torch
import numpy as np
import spconv
import torch.nn as nn
from functools import partial
import pvdet.dataset.utils.calibration as calibration
from pvdet.model.VoxelSA.voxel_sa_utils import VoxelSetAbstraction
import pvdet.model.pointnet2.pointnet2_stack.pointnet2_modules as pointnet2_stack_modules
import pvdet.model.pointnet2.pointnet2_stack.pointnet2_utils as pointnet2_stack_utils
from pvdet.ops.iou3d_nms import iou3d_nms_utils


def parse_config():
    args = edict()
    args.local_rank = 0
    args.batch_size = 1
    args.epochs = 80
    args.workers = 1
    args.extra_tag = "fix_iou3d_bug"
    args.ckpt = "/home/liang/PVRCNN-V1.1/ckpt/PVAnet/checkpoint_epoch_80.pth"
    args.pretrained_model = None

    args.ckpt_save_interval = 2
    args.max_ckpt_save_num = 40
    args.log_path = "/home/liang/PVRCNN-V1.1/for_ros/models"
    args.eval_all = False
    args.start_epoch = 0
    args.save_to_file = True
    args.split = "test"
    args.ckpt_id = 80

    return args


def parpare_point_cloud():
    path = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/kitti/testing/velodyne/000000.bin"
    points = np.fromfile(path,dtype=np.float32).reshape(-1,4)
    return points

def get_fov_flag(points, img_shape, calib):

        # 过滤得到前边90度范围内的点云
        # Valid point should be in the image (and in the PC_AREA_SCOPE)
        # :param pts_rect:
        # :param img_shape:
        # :return:
    pts_rect = calib.lidar_to_rect(points[:, 0:3])
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag

class PrepocessData:
    def __init__(self):
        self.voxel_generator_cfg = cfg.DATA_CONFIG.VOXEL_GENERATOR
        self.voxel_generator = spconv.utils.VoxelGeneratorV2(
        voxel_size=[0.05,0.05,0.1],
        point_cloud_range=[0,-40.0,-3.0,70.4,40.0,1.0],
        max_num_points=5,
        max_voxels=16000
    )
        self.image_shape = np.array([375,1242],dtype=np.int32)
        self.sparse_shape = self.voxel_generator.grid_size[::-1] + [1, 0, 0]

    def get_calib(self):
        calib_file = os.path.join("/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/kitti/testing/", 'calib', '000000.txt' )
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def points2voxel(self,points):
        voxel_grid = self.voxel_generator.generate(points)
        input_dict = {}

        input_dict["voxels"] = voxel_grid["voxels"]
        input_dict["coordinates"] = voxel_grid["coordinates"]
        input_dict["num_points"] = voxel_grid["num_points_per_voxel"]
        input_dict["voxel_centers"] = (input_dict["coordinates"][:, ::-1] + 0.5) * self.voxel_generator.voxel_size \
                        + self.voxel_generator.point_cloud_range[0:3]
        device = torch.cuda.current_device()
        input_dict["points"] = torch.tensor(points, dtype=torch.float32, device=device)
        input_dict["voxels"] = torch.tensor(input_dict["voxels"],dtype=torch.float32,device=device)
        input_dict["coordinates"] = torch.tensor(input_dict["coordinates"], dtype=torch.int32, device=device)
        input_dict["num_points"] = torch.tensor(input_dict["num_points"], dtype=torch.int32, device=device)
        input_dict["voxel_centers"] = torch.tensor(input_dict["voxel_centers"] , dtype=torch.float32, device=device)
        input_dict["image_shape"] = self.image_shape
        zeros_tensor = torch.zeros((input_dict["coordinates"].shape[0],1),dtype=torch.int32,device=device)
        input_dict["coordinates"] = torch.cat([zeros_tensor,input_dict["coordinates"]],dim=1)
        input_dict["points_mean"] = input_dict["voxels"][:, :, :].sum(dim=1, keepdim=False)\
                                    / input_dict["num_points"].type_as(input_dict["voxels"] ).view(-1, 1) #vfe
        input_dict["input_sp_tensor"] = spconv.SparseConvTensor(
            features=input_dict["points_mean"],
            indices=input_dict["coordinates"],
            spatial_shape=self.sparse_shape,
            batch_size=1
        )

        return input_dict


class DetNet(nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        #######################################稀疏三维卷积#########################################
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.downsample_times_map = [1, 2, 4, 8]
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        self.conv1 = spconv.SparseSequential(
            spconv.SparseSequential(
                spconv.SubMConv3d(16, 16, 3, padding=1, bias=False, indice_key='subm1'),
                norm_fn(16),
                nn.ReLU(),
            ))
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            spconv.SparseSequential(
                spconv.SparseConv3d(16, 32, 3, stride=2, padding=1,
                                    bias=False, indice_key='spconv2'),
                norm_fn(32),
                nn.ReLU(),
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(32, 32, 3, padding=1, bias=False, indice_key='subm2'),
                norm_fn(32),
                nn.ReLU(),
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(32, 32, 3, padding=1, bias=False, indice_key='subm2'),
                norm_fn(32),
                nn.ReLU(),
            )
        )
        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            spconv.SparseSequential(
                spconv.SparseConv3d(32, 64, 3, stride=2, padding=1,
                                    bias=False, indice_key='spconv3'),
                norm_fn(64),
                nn.ReLU(),
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(64, 64, 3, padding=1, bias=False, indice_key='subm3'),
                norm_fn(64),
                nn.ReLU(),
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(64, 64, 3, padding=1, bias=False, indice_key='subm3'),
                norm_fn(64),
                nn.ReLU(),
            )
        )
        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            spconv.SparseSequential(
                spconv.SparseConv3d(64, 64, 3, stride=2, padding=(0, 1, 1),
                                    bias=False, indice_key='spconv4'),
                norm_fn(64),
                nn.ReLU(),
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(64, 64, 3, bias=False, indice_key='subm4', padding=1),
                norm_fn(64),
                nn.ReLU(),
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(64, 64, 3, bias=False, indice_key='subm4'),
                norm_fn(64),
                nn.ReLU(),
            )
        )
        # last_pad = 0 if cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE[-1] in [0.1, 0.2] else (1, 0, 0)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=0,  # last_pad
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        #######################################接下来是RPN head #####################################
        BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        Conv2d = partial(nn.Conv2d, bias=False)########################################voxel_sa##########################################
        ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        self.block1 = nn.Sequential(
            nn.ZeroPad2d(1),
            Conv2d(256, 128, 3, stride=1),
            BatchNorm2d(128),
            nn.ReLU(),
                Conv2d(128,128,3,padding=1),#1
                BatchNorm2d(128),
                nn.ReLU(),
                Conv2d(128, 128, 3, padding=1),#2
                BatchNorm2d(128),
                nn.ReLU(),
                Conv2d(128, 128, 3, padding=1),#3
                BatchNorm2d(128),
                nn.ReLU(),
                Conv2d(128, 128, 3, padding=1),#4
                BatchNorm2d(128),
                nn.ReLU(),
                Conv2d(128, 128, 3, padding=1),#5
                BatchNorm2d(128),
                nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.ZeroPad2d(1),
            Conv2d(128, 256, 3, stride=2),
            BatchNorm2d(256),
            nn.ReLU(),
                Conv2d(256,256,3,padding=1),#1
                BatchNorm2d(256),
                nn.ReLU(),
                Conv2d(256, 256, 3, padding=1),#2
                BatchNorm2d(256),
                nn.ReLU(),
                Conv2d(256, 256, 3, padding=1),#3
                BatchNorm2d(256),
                nn.ReLU(),
                Conv2d(256, 256, 3, padding=1),#4
                BatchNorm2d(256),
                nn.ReLU(),
                Conv2d(256, 256, 3, padding=1),  #5
                BatchNorm2d(256),
                nn.ReLU(),
        )
        self.deblock1 = nn.Sequential(
            ConvTranspose2d(128, 256, 1,stride=1),
            BatchNorm2d(256),
            nn.ReLU()
        )
        self.deblock2 = nn.Sequential(
            ConvTranspose2d(256,256,2,stride=2),
            BatchNorm2d(256),
            nn.ReLU()
        )
        self.rpn_head_conv_cls =  nn.Conv2d(512, 6*3, 1)
        self.rpn_head_conv_box = nn.Conv2d(512, 6*7, 1)
        self.rpn_head_conv_dir_cls = nn.Conv2d(512,6*2,1)



        #######################################接下来是voxeel SA #####################################

        mlps = [[16, 16, 16], [16, 16, 16]]
        self.conv1_sa_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=[0.4, 0.8],
                nsamples=[16, 16],
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool',
            )
        self.conv2_sa_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=[0.8, 1.2],
            nsamples=[16, 32],
            mlps=[[32,32, 32], [32,32, 32]],
            use_xyz=True,
            pool_method='max_pool',
        )
        self.conv3_sa_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=[1.2, 2.4],
            nsamples=[16, 32],
            mlps=[[64,64, 64], [64, 64, 64]],
            use_xyz=True,
            pool_method='max_pool',
        )
        self.conv4_sa_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=[2.4, 4.8],
            nsamples=[16, 32],
            mlps=[[64,64, 64], [64,64, 64]],
            use_xyz=True,
            pool_method='max_pool',
        )
        c_in = 16+16 + 32+32 + 64+64 + 64+64 + 128*2 # C*D

        self.raw_sa_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=[0.4, 0.8],
            nsamples=[16, 16],
            mlps=[[1,16, 16], [1,16, 16]],
            use_xyz=True,
            pool_method='max_pool',
        )
        c_in += 16+16
        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, 128, bias=False), # c_in=640
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.num_point_features_before_fusion = c_in
        #############################################Point Head############################################
        self.point_head_cls_layer = nn.Sequential(
            nn.Linear(c_in,256,bias=False),#1
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256, bias=False),#2
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1, bias=True),#3
        )
        ###########################################RCNN########################################################
        self.roi_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=[0.8, 1.6],
            nsamples=[16, 16],
            mlps=[[128,64,64],[128,64,64]],
            use_xyz=True,
            pool_method="max_pool"
        )
        self.rcnn_share_layer = nn.Sequential(
            nn.Conv1d(6**3*(64+64),256,kernel_size=1,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256,256,kernel_size=1,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.rcnn_cls_layer = nn.Sequential(
            nn.Conv1d(256,256,kernel_size=1,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256,256,1,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,1,1,stride=1,bias=True)
        )
        self.rcnn_reg_layer = nn.Sequential(
            nn.Conv1d(256,256,1,stride=1,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256,256,1,stride=1,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,7,1,stride=1,bias=True)
        )
        self.voxel_size = [0.05,0.05,0.1]
        self.point_cloud_range =[0,-40.0,-3.0,70.4,40.0,1.0]
        self.anchors = self.get_anchor()


    def get_sampled_points(self,raw_points):
        # batch_size =1
        #对原始点云进行FPS采样，取2048个关键点
        keypoints_list = []
        sampled_points = raw_points[:,:3].unsqueeze(dim=0)
        cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
            sampled_points[:, :, 0:3].contiguous(), 2048
        ).long()
        if sampled_points.shape[1] < 2048:
            empty_num = 2048- sampled_points.shape[1]
            cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

        keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
        keypoints_list.append(keypoints)
        keypoints = torch.cat(keypoints_list, dim=0)
        return keypoints

    @staticmethod
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
        ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(
            torch.t(Id) * wd)
        return ans

    def interpolate_from_bev_features(self,keypoints,spatial_features):
        x_idxs = (keypoints[:, :, 0] - 0.0) / 0.05
        y_idxs = (keypoints[:, :, 1] - -40.0) / 0.05
        x_idxs = x_idxs / 8.0
        y_idxs = y_idxs / 8.0
        point_bev_features_list = []
        cur_x_idxs = x_idxs[0]
        cur_y_idxs = y_idxs[0]
        cur_bev_features = spatial_features[0].permute(1, 2, 0)
        point_bev_features = self.bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
        point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))
        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features


    def get_voxel_centers(self,voxel_coords, downsample_times, voxel_size, point_cloud_range):
        """
            Args:
                voxel_coords: (N, 3)
                downsample_times:
                voxel_size:
                point_cloud_range:
            Returns:
            """
        assert voxel_coords.shape[1] == 3
        voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
        voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
        pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
        voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
        return voxel_centers

    def get_anchor(self,):
        all_anchors = []
        num_anchors_per_location = []
        #anchor的一些参数，包括car，pedestrain，cyclist
        grid_sizes = [np.array([176, 200]), np.array([176, 200]), np.array([176, 200])]
        anchor_sizes = [[[1.6,3.9,1.56]],[[0.6,0.8,1.73]],[[0.6,1.76,1.73]]]
        anchor_rotations = [[0,1.57],[0,1.57],[0,1.57]]
        anchor_heights = [[-1.78],[-0.6],[-0.6]]
        align_center = [False,False,False]
        anchor_range = [0.0,-40.0,-3.0,70.4,40.0,1.0]

        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, anchor_sizes, anchor_rotations, anchor_heights, align_center):

            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            if align_center:
                x_stride = (anchor_range[3] - anchor_range[0]) / grid_size[0]
                y_stride = (anchor_range[4] - anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (anchor_range[3] - anchor_range[0]) / (grid_size[0] - 1)
                y_stride = (anchor_range[4] - anchor_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0

            x_shifts = torch.arange(
                anchor_range[0] + x_offset, anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            y_shifts = torch.arange(
                anchor_range[1] + y_offset, anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            z_shifts = x_shifts.new_tensor(anchor_height)

            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])  # [x_grid, y_grid, z_grid]
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [x, y, z, 3]
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat(
                [*anchors.shape[0:3], num_anchor_size, 1, 1])
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # [x, y, z, num_size, num_rot, 7]

            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            # anchors = anchors.view(-1, anchors.shape[-1])
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            all_anchors.append(anchors.cuda())
        return torch.cat(all_anchors,dim=-3).reshape(1,-1,7)

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza
        rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)

    @staticmethod
    def limit_period(val, offset=0.5, period=np.pi):
        assert val.dtype == torch.float32
        ans = val - torch.floor(val / period + offset) * period
        return  ans

    @staticmethod
    def rotate_points_along_z(points, angle):
        """
        Args:
            points: (B, N, 3 + C)
            angle: (B), angle along z-axis, angle increases x ==> y
        Returns:
        """
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa, sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot

    def forward(self,inputs):
        x = self.conv_input(inputs["input_sp_tensor"])
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        spatial_features = out.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)#压缩到二维特征

        #######################--------VOXEL_SA----------############################################################
        #对原始点云采集keypoints作为new_xyz
        keypoints = self.get_sampled_points(inputs["points"])
        point_features_list =[]
        point_bev_features = self.interpolate_from_bev_features(keypoints,spatial_features)
        point_features_list.append(point_bev_features)
        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)
        #对原始点云进行聚合
        xyz = inputs["points"][:,:3]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        xyz_batch_cnt[0] = xyz.shape[0]
        point_features = inputs["points"][:,3:]
        pooled_points, pooled_features = self.raw_sa_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )
        point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))
        #对conv1进行聚合
        cur_coords = x_conv1.indices
        xyz = self.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=self.downsample_times_map[0],
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        xyz_batch_cnt[0] = (cur_coords[:, 0] == 0).sum()
        pooled_points, pooled_features = self.conv1_sa_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=x_conv1.features.contiguous(),
        )
        point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))
        #对conv2聚合
        cur_coords = x_conv2.indices
        xyz = self.get_voxel_centers(
            cur_coords[:, 1:4],
            downsample_times=self.downsample_times_map[1],
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        xyz_batch_cnt[0] = (cur_coords[:, 0] == 0).sum()
        pooled_points, pooled_features = self.conv2_sa_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=x_conv2.features.contiguous(),
        )
        point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))
        # 对conv3聚合
        cur_coords = x_conv3.indices
        xyz = self.get_voxel_centers(
            cur_coords[:, 1:4],
            downsample_times=self.downsample_times_map[2],
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        xyz_batch_cnt[0] = (cur_coords[:, 0] == 0).sum()
        pooled_points, pooled_features = self.conv3_sa_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=x_conv3.features.contiguous(),
        )
        point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))
        # 对conv4聚合
        cur_coords = x_conv4.indices
        xyz = self.get_voxel_centers(
            cur_coords[:, 1:4],
            downsample_times=self.downsample_times_map[3],
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        xyz_batch_cnt[0] = (cur_coords[:, 0] == 0).sum()
        pooled_points, pooled_features = self.conv4_sa_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=x_conv4.features.contiguous(),
        )
        point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

        # 关键点的坐标：point_coords(bxyz)
        batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)
        #关键点的特征 :point_features
        point_features = torch.cat(point_features_list, dim=2)
        point_features_before_fusion = point_features.view(-1, point_features.shape[-1])#原始拼接特征
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))#经过线性层后的特征

        ############################################--RPN-Head----#################################################
        ups= []
        x = self.block1(spatial_features)
        ups.append(self.deblock1(x))
        x = self.block2(x)
        ups.append(self.deblock2(x))
        x = torch.cat(ups, dim=1)
        rpn_box_preds = self.rpn_head_conv_box(x)
        rpn_cls_preds = self.rpn_head_conv_cls(x)
        rpn_box_preds = rpn_box_preds.permute(0, 2, 3, 1).contiguous()
        rpn_cls_preds = rpn_cls_preds.permute(0, 2, 3, 1).contiguous()
        rpn_dir_preds = self.rpn_head_conv_dir_cls(x)
        #使用box回归值和anhcor进行decode,
        rpn_box_preds = rpn_box_preds.reshape(batch_size, -1, 7)
        rpn_cls_preds = rpn_cls_preds.reshape(batch_size, -1, 3)
        rpn_dir_preds_score = rpn_dir_preds.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
        dir_cls_preds_label= torch.argmax(rpn_dir_preds_score, dim=-1)
        rpn_box_preds = self.decode_torch(rpn_box_preds,self.anchors)
        rot_angle_preds = self.limit_period(rpn_box_preds[...,6]- 0.78539,offset=0.0,period=np.pi)
        rot_angle_preds_final = rot_angle_preds + 0.78539 + np.pi*dir_cls_preds_label.to(rpn_box_preds.dtype)
        rpn_box_preds[..., 6] = rot_angle_preds_final % (np.pi * 2)

        ##################################Point Head######################################################
        point_cls_preds = self.point_head_cls_layer(point_features_before_fusion)
        point_cls_score,_ = torch.sigmoid(point_cls_preds).max(dim=-1)
        ###################################-----RCNN--------##############################################
        #感兴趣区域的筛选
        rois = rpn_box_preds.new_zeros(batch_size,100,7)
        roi_score = rpn_box_preds.new_zeros(batch_size, 100)
        roi_labels = rpn_box_preds.new_zeros(batch_size, 100, dtype=torch.long)
        cur_box = rpn_box_preds[0]
        cur_cls = rpn_cls_preds[0]
        cur_roi_score, cur_roi_labels = torch.max(cur_cls, dim=1)
        rank_score, indices = torch.topk(cur_roi_score, k=min(1024, 7))#根据rpn类别得分得到前1024个box
        for_nms_box = cur_box[indices]
        nms_indices, _ = iou3d_nms_utils.nms_gpu(for_nms_box, rank_score, 0.7)#对1024个box进行nms
        selected = indices[nms_indices[:100]] #选则rpn类别得分前100个box进行second-stage细化
        rois[0,:len(selected),:] = cur_box[selected]
        roi_score[0, :len(selected)] = cur_roi_score[selected]
        roi_labels[0, :len(selected)] = cur_roi_labels[selected] + 1 #类别从1-3并非0-2，将0认为是背景应该是
        #将感兴趣区域网格化，每个3d-box转化为6*6*6的网格
        rois = rois.view(-1,7)
        local_grid = rois.new_ones(6, 6, 6)
        local_grid_id = torch.nonzero(local_grid).float()#(6*6*6,3)
        local_grid_id = local_grid_id.repeat(100, 1, 1)#(100,216,3)
        rois_size = rois[..., 3:6].unsqueeze(dim=1).clone()
        gloal_grid_points = (local_grid_id + 0.5) * (rois_size / 6) - rois_size / 2 #将原点转化到box中心
        rois_ry = rois[..., 6].clone()
        gloal_grid_points = self.rotate_points_along_z(
            gloal_grid_points.clone(), rois_ry.view(-1))#将网格化的rois由规定坐标系旋转到真实坐标系
        roi_centers = rois[..., 0:3].clone()
        gloal_grid_points += roi_centers.unsqueeze(dim=1)#平移网格化的roi
        gloal_grid_points = gloal_grid_points.view(batch_size, -1, gloal_grid_points.shape[-1])
        #网格化的roi作为关键点，聚合keypoints的特征
        xyz = point_coords[:,1:4]
        xyz_batch_count = xyz.new_zeros(batch_size, ).int()
        xyz_batch_count[0] = keypoints.shape[1]#keypoints的个数
        new_xyz = gloal_grid_points.view(-1, 3)
        new_xyz_bn_count = new_xyz.new_zeros(batch_size).int()
        new_xyz_bn_count[0] = gloal_grid_points.shape[1]
        pooled_xyz, pooled_features = self.roi_pool_layer(
            xyz.contiguous(),
            xyz_batch_count,
            new_xyz,
            new_xyz_bn_count,
            features=point_features
        )
        pooled_features = pooled_features.view(-1, 6 ** 3, pooled_features.shape[-1])
        pooled_features = pooled_features.permute(0, 2, 1).contiguous()  # (100,128,6*6*6)
        share_features = self.rcnn_share_layer(pooled_features.view(100, -1, 1))
        rcnn_cls_preds = self.rcnn_cls_layer(share_features).squeeze(dim=-1).contiguous()
        rcnn_reg_preds = self.rcnn_reg_layer(share_features).squeeze(dim=-1).contiguous()
        rcnn_cls_preds = rcnn_cls_preds.unsqueeze(dim=0)
        rcnn_reg_preds = rcnn_reg_preds.unsqueeze(dim=0)
        #将second-stage阶段的回归结果根据roi进行解码，得到最终的box,解码顺序就是先不带xyz解码，然后在roi的xyz加上
        rois = rois.unsqueeze(dim=0)
        rois_y = rois[..., 6].view(-1)
        rois_xyz = rois[..., 0:3]
        rois_temp = rois.clone().detach()
        rois_temp[..., 0:3] = 0
        box_ct = self.decode_torch(rcnn_reg_preds, rois_temp).view(-1, 7)
        box_preds = self.rotate_points_along_z(
            box_ct.unsqueeze(dim=1), rois_y).squeeze(dim=1).view(batch_size, -1, 7)
        box_preds[..., 0:3] += rois_xyz
        #最终的输出为：box_preds，rcnn_cls_preds
        #############################################----POST-Processing----######################################
        cur_box_preds = box_preds[0]
        cur_cls_preds = rcnn_cls_preds[0]
        cur_normal_cls_preds = torch.sigmoid(cur_cls_preds)
        label_preds = roi_labels[0]
        #进行后处理
        score_mask = ( cur_normal_cls_preds>0.1).squeeze(dim=-1)
        pps_box_scores = cur_normal_cls_preds[score_mask]#post processing box
        pps_box_preds = cur_box_preds[score_mask]
        selected = []
        if pps_box_scores.shape[0] > 0:
            score_for_nms, indices = torch.topk(
                pps_box_scores.squeeze(dim=-1), k=min(4096, pps_box_scores.shape[0]))
            box_for_nms = pps_box_preds[indices]
            keep_id, _ = iou3d_nms_utils.nms_gpu(box_for_nms, score_for_nms,0.1)
            selected = indices[keep_id[:500]]
            origin_idx = torch.nonzero(score_mask).view(-1)
            selected = origin_idx[selected].view(-1)
        final_box = cur_box_preds[selected]
        final_cls = label_preds[selected]
        final_score = cur_normal_cls_preds[selected]
        return (final_box,final_cls,final_score)


    def load_params_from_file(self,filename,logger,to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        if logger is not None:
            logger.info("****Load paramters from checkpoint %s to %s" % (filename,"CPU" if to_cpu else "GPU"))
        loc_type = torch.device("cpu") if to_cpu else None
        checkpoint = torch.load(filename,map_location=loc_type)
        model_state_disk = checkpoint["model_state"]
        for key,val in model_state_disk.items():
            logger.info(key)

        if "version" in checkpoint:
            logger.info("===>checkpoint trained from version:%s"%checkpoint["version"])

        update_model_state={}
        state_dict = self.state_dict()
        current_model_state_key = [cur_key for cur_key,vals in state_dict.items()]
        if len(model_state_disk) == len(state_dict):
            print("参数已经对齐")
        else:
            print("len(model_state_disk)",len(model_state_disk))
            print("len(state_dict)",len(state_dict))
        for (key,val),i in zip(model_state_disk.items(),range(len(current_model_state_key))):
            # if key in self.state_dict() and self.state_dict()[key].shape==model_state_disk[key].shape:
            key = current_model_state_key[i]
            if state_dict[key].shape == val.shape:
                update_model_state[key]= val
            else:
                raise ModuleNotFoundError
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info("Not update weight %s: %s"%(key,str(state_dict[key].shape)))
        logger.info("==>Done (load %d/%d)"%(len(update_model_state),len(model_state_disk.items())))

class DetNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,3,1,padding=1)
    def forward(self,x):
        x = self.conv1(x)
        return x




def origin_model():
    args = parse_config()
    ckpt_dir = args.ckpt
    log_file = os.path.join(args.log_path, "log_eval_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = common_utils.create_logger(log_file)
    prepocess_model = PrepocessData()
    test_set,test_dataloader,sampler = build_data_loader(
        dist=False,
        data_path=cfg.DATA_DIR,
        batch_size=args.batch_size,
        num_workers=cfg.DATA_CONFIG.NUM_WORKERS,
        logger=None,
        split=args.split,
        training=False
    )
    model = build_netword(num_class=len(cfg.CLASS_NAMES),dataset=test_set,logger=logger)
    model.load_params_from_file(filename=ckpt_dir, logger=logger)
    with torch.no_grad():
        test_dataloader_iter = iter(test_dataloader)
        model.cuda()
        model.eval()
        data = next(test_dataloader_iter)
        input_dict = example_convert_to_torch(data)
        points = parpare_point_cloud()#取出点云
        fov_flag = get_fov_flag(points,prepocess_model.image_shape,prepocess_model.get_calib())#过滤点云
        points = points[fov_flag]
        inputs = prepocess_model.points2voxel(points)

        pred_dict, ret_dict = model(input_dict)
        # script_model = torch.jit.trace(model, input_dict)
    print("done")

def save_model():
    args = parse_config()
    ckpt_dir = args.ckpt
    log_file = os.path.join(args.log_path, "log_eval_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = common_utils.create_logger(log_file)
    prepocess_model = PrepocessData()
    model = DetNet(4)
    model.load_params_from_file(filename=ckpt_dir, logger=logger)
    points = parpare_point_cloud()
    fov_flag = get_fov_flag(points, prepocess_model.image_shape, prepocess_model.get_calib())
    points = points[fov_flag]
    with torch.set_grad_enabled(False):
        inputs = prepocess_model.points2voxel(points)
        model.cuda()
        model.eval()
        output = model(inputs)
    # torch.save(model,"DetNet.pkl")
    print("done")


def detect():
    prepocess_model = PrepocessData()
    model = torch.load("DetNet.pkl")
    points = parpare_point_cloud()
    fov_flag = get_fov_flag(points, prepocess_model.image_shape, prepocess_model.get_calib())
    points = points[fov_flag]
    with torch.set_grad_enabled(False):
        input_sp_tensor = prepocess_model.points2voxel(points)
        model.cuda()
        model.eval()
        output = model(input_sp_tensor)
    print(output)



if __name__ == '__main__':
    save_model()
    # main()
    # main3()

