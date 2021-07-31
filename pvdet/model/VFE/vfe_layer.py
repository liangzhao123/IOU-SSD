import torch
import torch.nn as nn
from pvdet.tools.config import cfg

class VoxelFeatureExtractor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError

class MeanVoxelFeatureExtractor(VoxelFeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__()

    def get_output_feature_dim(self):
        return cfg.DATA_CONFIG.NUM_POINT_FEATURES['use']

    def forward(self, features, num_voxels, **kwargs):
        """
        用于取每个体素内点的平均值
        :param features: (N, max_points_of_each_voxel, 3 + C)
        :param num_voxels: (N)
        :param kwargs:
        :return:
        """
        points_mean = features[:, :, :].sum(dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()



