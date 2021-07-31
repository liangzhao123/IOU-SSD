import torch
import spconv
import torch.nn as nn


class Net_3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=None
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None
    def forward(self,batch_data):
        voxel_features = batch_data["voxels"]
        x1 = self.conv1(voxel_features)