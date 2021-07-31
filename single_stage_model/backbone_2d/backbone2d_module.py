import torch.nn as nn
import torch
from functools import partial
def Layer(in_channels,
          out_channels,
          kernel_size,
          stride,
          padding=0,
          bias=True,conv_type="conv2d"):
    BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
    Conv2d = partial(nn.Conv2d, bias=False)
    ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
    if conv_type=="conv2d":
        m = nn.Sequential(
            Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias,),
            BatchNorm2d(out_channels),
            nn.ReLU()
        )
    elif conv_type=="deconv2d":
        m = nn.Sequential(
            ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias, ),
            BatchNorm2d(out_channels),
            nn.ReLU()
        )
    else:
        raise NotImplementedError
    return m

class Backbone2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.padding_layer = nn.ZeroPad2d(1)
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d(1),
            Layer(512,256,3,stride=1,bias=False),
            Layer(256,256,3,stride=1,padding=1,bias=False),
            Layer(256,256,3,stride=1,padding=1,bias=False),
            Layer(256,256,3, stride=1, padding=1,bias=False),
            Layer(256, 256, 3, stride=1, padding=1,bias=False),
            Layer(256, 256, 3, stride=1, padding=1,bias=False),
            Layer(256, 256, 3, stride=1, padding=1,bias=False),
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d(1),
            Layer(256,512,3,stride=2,bias=False),
            Layer(512,512,3,stride=1,padding=1,bias=False),
            Layer(512,512,3,stride=1,padding=1,bias=False),
            Layer(512, 512, 3, stride=1, padding=1,bias=False),
            Layer(512, 512, 3, stride=1, padding=1,bias=False),
            Layer(512, 512, 3, stride=1, padding=1,bias=False),
            Layer(512, 512, 3, stride=1, padding=1,bias=False),
        )
        self.deconv1 = nn.Sequential(
            Layer(256,512,1,stride=1,bias=False,conv_type="deconv2d"),
        )
        self.deconv2 = nn.Sequential(
            Layer(512, 512, 2, stride=2, bias=False, conv_type="deconv2d"),
        )

    def forward(self,batch_dict):
        spatial_feartures = batch_dict["spatial_features"]
        ups = []
        conv_last = []
        x = self.conv1(spatial_feartures)
        conv_last = x
        ups.append(self.deconv1(x))
        x = self.conv2(x)
        ups.append(self.deconv2(x))
        x = torch.cat(ups, dim=1)

        batch_dict["conv2d_features"] = x
        batch_dict["conv2d_last_features"] = conv_last
        return batch_dict
