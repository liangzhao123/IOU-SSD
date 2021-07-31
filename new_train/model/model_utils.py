# from new_train.config import cfg
#
#
# import torch
# import torch.nn as nn
# import spconv
# from functools import partial
# from pvdet.dataset.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
# import numpy as np
# from pvdet.tools.utils import loss_utils
#
#
#
# class VEF(nn.Module):
#     def __init__(self,num_channel):
#         super().__init__()
#         self.used_feature = num_channel
#     def forward(self,voxels,num_per_voxel):
#         mean_voxel = voxels.sum(dim=1)/ num_per_voxel.type_as(voxels).view(-1,1)
#         return mean_voxel.contiguous()
#
# class Conv_3d_net(nn.Module):
#     def __init__(self,in_channels):
#         super().__init__()
#         norm_fn = partial(nn.BatchNorm1d,eps=1e-3, momentum=0.01)
#         self.conv_input = spconv.SparseSequential(
#             spconv.SubMConv3d(in_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
#             norm_fn(16),
#             nn.ReLU(),
#         )
#
#         block = self.post_act_block
#
#         self.conv1 = spconv.SparseSequential(
#             block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
#         )
#
#         self.conv2 = spconv.SparseSequential(
#             # [1600, 1408, 41] <- [800, 704, 21]
#             block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
#             block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
#             block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
#         )
#
#         self.conv3 = spconv.SparseSequential(
#             # [800, 704, 21] <- [400, 352, 11]
#             block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
#         )
#
#         self.conv4 = spconv.SparseSequential(
#             # [400, 352, 11] <- [200, 176, 5]
#             block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
#         )
#
#         last_pad = 0 if cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE[-1] in [0.1, 0.2] else (1, 0, 0)
#
#         self.conv_out = spconv.SparseSequential(
#             # [200, 150, 5] -> [200, 150, 2]
#             spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
#                                 bias=False, indice_key='spconv_down2'),
#             norm_fn(128),
#             nn.ReLU(),
#         )
#         self.seg_net = SegNet(16)
#         self.seg_loss_func =loss_utils.SigmoidFocalClassificationLoss_v1(alpha=0.25, gamma=2.0)
#         self.num_point_features = 128
#         self.ret = {}
#
#     def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0,
#                        conv_type='subm', norm_fn=None):
#         if conv_type == 'subm':
#             m = spconv.SparseSequential(
#                 spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key),
#                 norm_fn(out_channels),
#                 nn.ReLU(),
#             )
#         elif conv_type == 'spconv':
#             m = spconv.SparseSequential(
#                 spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
#                                     bias=False, indice_key=indice_key),
#                 norm_fn(out_channels),
#                 nn.ReLU(),
#             )
#         elif conv_type == 'inverseconv':
#             m = spconv.SparseSequential(
#                 spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size,
#                                            indice_key=indice_key, bias=False),
#                 norm_fn(out_channels),
#                 nn.ReLU(),
#             )
#         else:
#             raise NotImplementedError
#         return m
#     def get_seg_target(self,voxel_center,coordinates,gt_boxes_lidar):
#         n = voxel_center.shape[0]#voxel per frame
#         one_hot_seg_target = torch.zeros((n, 2), dtype=torch.float32).cuda()
#         seg_label_batch = []
#         for bs in range(self.batch_size):
#             cur_voxels = (voxel_center[coordinates[:,0]==bs]).cpu()
#             cur_gt_boxes = torch.tensor(gt_boxes_lidar[bs],dtype=torch.float64)
#             seg_label = points_in_boxes_cpu(cur_voxels,cur_gt_boxes[:,:7])
#             seg_label = seg_label.sum(dim = 0)
#             seg_label = torch.where(seg_label>torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0)).cuda().type(torch.float32)
#             seg_label_batch.append(seg_label)
#             one_hot_seg_target[coordinates[:,0]==bs] = one_hot_seg_target[coordinates[:,0]==bs].scatter_(-1,seg_label.unsqueeze(dim=-1).long(),1.0)
#         seg_label_batch = torch.cat(seg_label_batch,dim=0).cuda()
#         self.ret["seg_label_src"] = seg_label_batch.unsqueeze(dim=0)
#         return one_hot_seg_target
#
#     def get_loss(self,coordinate):
#         batch_size = self.ret["batch_size"]
#         seg_label_src = self.ret["seg_label_src"]
#         postives = seg_label_src>0
#         negitive = seg_label_src==0
#         cls_target = self.ret["seg_target"].unsqueeze(dim=0)
#         cls_pred = self.ret["seg_pred"].unsqueeze(dim=0)
#         cls_weight = postives.float()*1.0+negitive.float()*1.0
#         postives_normal = postives.sum(1,keepdim=True).float()
#         cls_weight /= torch.clamp(postives_normal,min=1)
#         seg_loss_src = self.seg_loss_func(cls_pred,cls_target,cls_weight)
#         seg_loss = seg_loss_src.sum()/batch_size
#         seg_loss = seg_loss*cfg.MODEL.CONV3D["seg_loss_weight"]
#         return seg_loss
#
#
#
#     def forward(self,input_sp_tensor,
#                 voxel_centers,
#                 coordinates,
#                 gt_boxes):
#         """
#                 :param voxel_features:  (N, C)
#                 :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
#                 :param batch_size:
#                 :return:
#                 """
#
#         x = self.conv_input(input_sp_tensor)#[41,1600,1408]
#
#         x_conv1 = self.conv1(x) #[41,1600,1408]
#         x_conv2 = self.conv2(x_conv1)#[21,800,704]
#         x_conv3 = self.conv3(x_conv2)#[11,400,352]
#         x_conv4 = self.conv4(x_conv3)#[5,200,176]
#
#         # for detection head
#         # [200, 176, 5] -> [200, 176, 2]
#         out = self.conv_out(x_conv4)#[2,200,176]
#         spatial_features = out.dense()
#         N, C, D, H, W = spatial_features.shape# batch size,128,2,200,176
#         spatial_features = spatial_features.view(N,C*D,H,W) #[batch_size,128*2,200,176]
#         seg_pred = self.seg_net(x_conv1)
#         self.ret.update({"spatial_features": spatial_features,
#                          "multi_scla_features": {"conv3d_1": x_conv1,
#                                                  "conv3d_2": x_conv2,
#                                                  "conv3d_3": x_conv3,
#                                                  "conv3d_4": x_conv4},
#                          "seg_pred": seg_pred,
#                          "batch_size": N})
#         if self.training:
#             self.batch_size = N
#             seg_target = self.get_seg_target(voxel_centers,
#                                              coordinates,
#                                              gt_boxes)#(batch_size*n,n)
#             self.ret["seg_target"] = seg_target
#             # loss = self.get_loss(batch_dict["coordinates"])
#         return self.ret
#
#
#
# class SegNet(nn.Module):
#     def __init__(self,inchannel):
#         super().__init__()
#         norm_fun = partial(nn.BatchNorm1d,eps=1e-3,momentum=0.01)
#         block = self.conv_block
#
#         self.conv1 = nn.Sequential(nn.Conv1d(inchannel,32,1,bias=False,stride=1),
#                                    norm_fun(32),
#                                    nn.ReLU())
#         self.conv2 = nn.Sequential(nn.Conv1d(32,32,1,bias=False,stride=1),
#                                    norm_fun(32),
#                                    nn.ReLU())
#         self.conv3 = nn.Sequential(nn.Conv1d(32,32,1,bias=False,stride=1),
#                                    norm_fun(32),
#                                    nn.ReLU())
#         self.deconv1  = nn.Sequential(nn.ConvTranspose1d(32,64,1,stride=1,bias=False),
#                                      norm_fun(64),
#                                      nn.ReLU())
#         self.conv4 = nn.Sequential(nn.Conv1d(32,32,1,bias=False,stride=1),
#                                    norm_fun(32),
#                                    nn.ReLU())
#         self.conv5 = nn.Sequential(nn.Conv1d(32,32,1,bias=False,stride=1),
#                                    norm_fun(32),
#                                    nn.ReLU())
#         self.conv6 = nn.Sequential(nn.Conv1d(32,32,1,bias=False,stride=1),
#                                    norm_fun(32),
#                                    nn.ReLU())
#         self.deconv2 = nn.Sequential(nn.ConvTranspose1d(32,64,1,stride=1,bias=False),
#                                      norm_fun(64),
#                                      nn.ReLU())
#
#         self.cls_layer = nn.Conv1d(128,2,1,bias=True)
#
#         self.seg_loss_func = loss_utils.SigmoidFocalClassificationLoss_v1(
#             alpha=0.25, gamma=2.0)
#         self.ret = {}
#         self.init_weights()
#     def init_weights(self):
#         pi = 0.01
#         nn.init.constant_(self.cls_layer.bias, -np.log((1 - pi) / pi))
#
#     def conv_block(self,key,inchannel,outchannel,kernel_size,stride,padding,bias=False):
#         norm_fun = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
#         conv = None
#         if key =="conv1d":
#             norm_fun = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
#             conv = nn.Sequential(nn.Conv1d(inchannel,outchannel,kernel_size,stride=stride,padding=padding,bias=bias),
#                                  norm_fun(outchannel),
#                                  nn.ReLU())
#         elif key == "ConvTranspose1d":
#             conv = nn.Sequential(
#                 nn.ConvTranspose1d(inchannel, outchannel, kernel_size, stride=stride, padding=padding, bias=bias),
#                 norm_fun(outchannel),
#                 nn.ReLU())
#         return conv
#
#
#     def forward(self,x_conv1):
#         x_input = x_conv1.features.view(-1,16).permute(1,0).unsqueeze(dim=0)#16000
#         ups = []
#         x = self.conv1(x_input)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         ups.append(self.deconv1(x))
#
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.conv6(x)
#         ups.append(self.deconv2(x))
#
#         x = torch.cat(ups,dim=1)
#         seg_pred = self.cls_layer(x)
#
#
#         return seg_pred.view(-1,2)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# if __name__ == '__main__':
#     pass
#
#
# # if __name__ == '__main__':
# #     zeors = torch.zeros((1,2),dtype=torch.float32)
# #     values = torch.tensor([[1]],dtype=torch.float32)
# #     one_hot = zeors.scatter(-1,values.long(),1)
# #     print("done")
# #x [2,16,16000]->[2,32,8000]->[2,32,8000]->[2,32,8000]->
# #ups[2,64,15999]
