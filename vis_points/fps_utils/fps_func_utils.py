from torch.autograd  import Function
import fps_with_features_cuda
import torch

class FPS_F(Function):
    @staticmethod
    def forward(ctx,xyz:torch.tensor,predix:torch.tensor,npoints:int):
        # xyz is not only has coordinate, xyz = [x,y,z,c1,c2...,cn] shape(B,N,C)
        assert xyz.is_contiguous()
        B,N,C = xyz.size()
        m1 = predix.shape[0]
        output = torch.cuda.IntTensor(B,npoints)
        temp = torch.cuda.FloatTensor(B,N).fill_(1e10)
        fps_with_features_cuda.fps_with_features_wrapper(B, N, npoints,m1, C,xyz,predix, temp, output)
        return output
    @staticmethod
    def backward(xyz, a=None):
        return None,None
fps_with_featres = FPS_F.apply

class FPS_D(Function):
    @staticmethod
    def forward(ctx,xyz:torch.tensor,npoint:int):
        # predix is the index of forground point (B,m2,1)
        #xyz (B,N,3)
        assert xyz.is_contiguous()
        B, N, C = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e18)
        fps_with_features_cuda.furthest_point_sampling_wrapper(B, N, npoint,xyz, temp, output)
        return output
    def backward(ctx, a = None):
        return None,None
fps_with_distance = FPS_D.apply