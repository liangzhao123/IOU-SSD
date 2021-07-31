import torch.nn as nn
import spconv
from functools import partial
import time
class Backbone3d(nn.Module):
    def __init__(self,in_channel,config):
        super().__init__()
        self.print_time = False
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channel,16,3,bias=False,padding=1,indice_key="subm1"),
            norm_fn(16),
            nn.ReLU(),
        )
        self.config = config

        block = self.convblock

        if self.config["add_layers_for_conv3d"]:
            self.conv1 = spconv.SparseSequential(
                block(16, 16, 3, padding=1, norm_fn=norm_fn, indice_key="subm1"),
                block(16, 16, 3, padding=1, norm_fn=norm_fn, indice_key="subm1"),
            )
            self.conv2 = spconv.SparseSequential(
                block(16, 32, 3, stride=2, padding=1, norm_fn=norm_fn, conv_type="spconv", indice_key="spconv2"),
                block(32, 32, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm2"),
                block(32, 32, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm2"),
                block(32, 32, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm2"),
                block(32, 32, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm2"),
            )
            self.conv3 = spconv.SparseSequential(
                block(32, 64, 3, stride=2, padding=1, norm_fn=norm_fn, conv_type="spconv", indice_key="spconv3"),
                block(64, 64, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm3"),
                block(64, 64, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm3"),
                block(64, 64, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm3"),
                block(64, 64, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm3"),
            )
            self.conv4 = spconv.SparseSequential(
                block(64, 128, 3, stride=2, padding=1, norm_fn=norm_fn, conv_type="spconv", indice_key="spconv4"),
                block(128, 128, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm4"),
                block(128, 128, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm4"),
                block(128, 128, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm4"),
                block(128, 128, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm4"),
            )
            self.conv_out = spconv.SparseSequential(
                block(128, 256, (3,1,1), stride=(2,1,1), padding=0, norm_fn=norm_fn, conv_type="spconv", indice_key="conv_down"),
                block(256, 256, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm5"),
                block(256, 256, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm5"),
                block(256, 256, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm5"),
            )
        else:
            self.conv1 = spconv.SparseSequential(
                block(16, 16, 3, padding=1, norm_fn=norm_fn, indice_key="subm1")
            )
            self.conv2 = spconv.SparseSequential(
                block(16, 32, 3, stride=2, padding=1, norm_fn=norm_fn, conv_type="spconv", indice_key="spconv2"),
                block(32, 32, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm2"),
                block(32, 32, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm2"),
                block(32, 32, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm2"),
            )
            self.conv3 = spconv.SparseSequential(
                block(32, 64, 3, stride=2, padding=1, norm_fn=norm_fn, conv_type="spconv", indice_key="spconv3"),
                block(64, 64, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm3"),
                block(64, 64, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm3"),
                block(64, 64, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm3"),
            )
            self.conv4 = spconv.SparseSequential(
                block(64, 128, 3, stride=2, padding=1, norm_fn=norm_fn, conv_type="spconv", indice_key="spconv4"),
                block(128, 128, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm4"),
                block(128, 128, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm4"),
                block(128, 128, 3, stride=1, padding=1, norm_fn=norm_fn, conv_type="subm", indice_key="subm4"),
            )
            self.conv_out = spconv.SparseSequential(
                spconv.SparseConv3d(128, 256, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=0,
                                    indice_key="conv_down"),
                norm_fn(256),
                nn.ReLU()
            )
        self.ret = {}

    def convblock(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0,
                       conv_type='subm', norm_fn=None):
        if conv_type=="subm":
            m = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels,out_channels,kernel_size,stride,
                                  padding,bias=False,indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU()
            )
        elif conv_type =="spconv":
            m = spconv.SparseSequential(
                spconv.SparseConv3d(in_channels,out_channels,kernel_size,stride,
                                    padding,indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU()
            )
        elif conv_type=="inverseconv":
            m = spconv.SparseSequential(
                spconv.SparseInverseConv3d(in_channels,out_channels,kernel_size,
                                           indice_key=indice_key,bias=False),
                norm_fn(out_channels),
                nn.ReLU()
            )
        else:
            raise NotImplementedError
        return m

    def weight_init(self):
        pass
    def target_assigner(self):
        pass
    def get_loss(self):
        pass
    def forward(self,input_sp_tensor,
                batch_data):
        voxel_input = batch_data["voxels"]
        if self.print_time:
            start = time.time()
        conv_input = self.conv_input(input_sp_tensor) #[41,1600,1408]
        if self.print_time:
            print("conv_input spend time:",time.time()-start)
        if self.print_time:
            start = time.time()
        conv_1 = self.conv1(conv_input)#[21,1600,1408]
        if self.print_time:
            print("conv1 spend time:",time.time()-start)
        if self.print_time:
            start = time.time()
        conv_2 = self.conv2(conv_1)#[21,800,704]
        if self.print_time:
            print("conv2 spend time:",time.time()-start)
        if self.print_time:
            start = time.time()
        conv_3 = self.conv3(conv_2)#[11,400,352]
        if self.print_time:
            print("conv3 spend time:",time.time()-start)
        if self.print_time:
            start = time.time()
        conv_4 = self.conv4(conv_3)#[6,200,176]
        if self.print_time:
            print("conv4 spend time:",time.time()-start)
        out = self.conv_out(conv_4)#[2,200,176]

        spatial_features = out.dense()
        N, C, D, H, W = spatial_features.shape  # batch size,256,2,200,176
        spatial_features = spatial_features.view(N, C * D, H, W)  # [batch_size,256*2,200,176]

        self.ret = {"spatial_features":spatial_features,
                    "voxel_features":{
                        "conv_1": conv_1,
                        "conv_2": conv_2,
                        "conv_3": conv_3,
                        "conv_4": conv_4,
                    },
                    }
        batch_data.update(self.ret)
        return batch_data