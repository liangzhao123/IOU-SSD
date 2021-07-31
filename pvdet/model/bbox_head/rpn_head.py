import torch
import numpy as np
import torch.nn as nn
from functools import partial
from pvdet.model.model_utils.pytorch_utils import Empty,Sequential

from pvdet.tools.config import cfg
from pvdet.model.bbox_head.anchor_target import AxisAlignedTargetAssigner,AnchorGeneratorRange
from pvdet.tools.utils import box_coder_utils
from pvdet.tools.utils import loss_utils
from pvdet.dataset.utils import common_utils
from pvdet.model.bbox_head.anchor_generator import AnchorGenertor

class AnchorHead(nn.Module):
    def __init__(self,grid_size,anchor_target_cfg):
        super().__init__()


        self.class_names = cfg.CLASS_NAMES
        self.num_class = len(cfg.CLASS_NAMES)
        self.forward_ret_dict = None
        #self.num_anchors_per_location = anchor_target_cfg.num_anchors_per_location
        #编码函数
        self.box_coder = box_coder_utils.ResidualCoder_v1()
        self.use_multihead = False
        self.box_code_size = self.box_coder.code_size
        #target生成函数
        self.target_assigner = AxisAlignedTargetAssigner(
                anchor_cfg=anchor_target_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
            )
        anchors, self.num_anchors_per_location = self.get_anchor(grid_size,anchor_target_cfg.ANCHOR_GENERATOR)
        self.anchors = [x.cuda() for x in anchors]
        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        self.build_losses(cfg.MODEL.RPN.RPN_HEAD.LOSSES)

    def get_anchor(self,grid_size,anchor_target_cfg):
        point_cloud_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
        anchor_generator_config = cfg.MODEL.RPN.RPN_HEAD.TARGET_CONFIG.ANCHOR_GENERATOR
        anchor_generator = AnchorGenertor(point_cloud_range,
                                          anchor_generator_config)
        features_map_size = [grid_size[:2]//config["feature_map_stride"] for
                             config in anchor_generator_config ]
        anchors_list, num_anchors_per_location_list = anchor_generator.generator(features_map_size)
        return anchors_list, num_anchors_per_location_list
    def get_assigner_target(self,gt_boxes):
        """
                :param gt_boxes: (B, N, 8)
                :return:
                """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes, self.use_multihead
        )

        return targets_dict
    def build_losses(self,losses_cfg):
        # loss function definition
        self.cls_loss_layer = loss_utils.SigmoidFocalClassificationLoss_v1(alpha=0.25, gamma=2.0)
        self.reg_loss_layer = loss_utils.WeightedSmoothL1Loss_v1(code_weights=losses_cfg['code_loss_weight'])
        self.dir_loss_layer = loss_utils.WeightedCrossEntropyLoss_v1()

    def get_reg_loss(self,):
        box_preds = self.forward_ret_dict["box_preds"]

        box_reg_targets = self.forward_ret_dict['box_reg_targets']

        #计算权重
        positives = (self.forward_ret_dict["box_cls_labels"]>0).float()
        positive_normal = torch.clamp(positives.sum(1,keepdim=True),min=1.0)
        reg_weights = positives.float()/positive_normal
        reg_weights_np= reg_weights.cpu().numpy()

        batch_size = box_preds.shape[0]
        box_preds = box_preds.reshape(batch_size,-1,box_reg_targets.shape[-1])
        #sin(a-b) = sina*cosb-cosa*sinb
        box_preds_with_sin,box_reg_targets_with_sin = self.add_sin_differece(box_preds,box_reg_targets)

        reg_loss_src = self.reg_loss_layer(box_preds_with_sin,box_reg_targets_with_sin,weights=reg_weights)
        reg_loss = reg_loss_src.sum()/batch_size
        reg_loss = reg_loss*cfg.MODEL.RPN.RPN_HEAD.LOSSES["reg_loss_weight"]
        tb_dict = {"rpn_reg_loss":reg_loss.item()}
        return reg_loss,tb_dict
    def add_sin_differece(self,box_a,box_b,dim = 6):
        box_a_sin_dif = torch.sin(box_a[...,dim])*torch.cos(box_b[...,dim])
        box_b_sin_dif = torch.cos(box_a[...,dim])*torch.sin(box_b[...,dim])
        box_a = torch.cat([box_a[...,:dim],box_a_sin_dif.unsqueeze(dim=-1)],dim=-1)
        box_b = torch.cat([box_b[...,:dim],box_b_sin_dif.unsqueeze(dim=-1)],dim=-1)
        return box_a,box_b
    def get_cls_loss(self):
        box_cls = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = box_cls.shape[0]
        cared = box_cls_labels>=0
        postives = box_cls_labels>0
        negtives = box_cls_labels==0
        cls_weights = negtives.float() *1.0 + postives.float()*1.0
        #box_cls_labels_np = box_cls_labels.cpu().numpy()
        # print(cls_weights.max(),cls_weights.min())
        box_cls_labels = box_cls_labels*cared.type_as(box_cls_labels)
        #box_cls_labels_np = box_cls_labels.cpu().numpy()
        box_cls = box_cls.reshape(batch_size,-1,self.num_class)
        one_hot_cls_targets = torch.zeros(*list(box_cls_labels.shape),self.num_class+1,
                                          dtype=box_cls_labels.dtype,device=box_cls_labels.device).cuda()
        one_hot_cls_targets.scatter_(-1,box_cls_labels.unsqueeze(dim=-1).long(),1.0)
        one_hot_cls_targets = one_hot_cls_targets[...,1:]
        positive_normal = postives.sum(1,keepdim=True).float()

        cls_weights /= torch.clamp(positive_normal,min=1)
        cls_loss_src = self.cls_loss_layer(box_cls,one_hot_cls_targets,weights=cls_weights)
        cls_loss = cls_loss_src.sum()/batch_size
        cls_loss = cls_loss * cfg.MODEL.RPN.RPN_HEAD.LOSSES["cls_weight"]
        tb_dict = {"rpn_cls_loss":cls_loss.item()}

        return cls_loss,tb_dict


    def get_dir_loss(self):
        box_preds = self.forward_ret_dict["box_preds"]
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = box_preds.shape[0]
        postives = (box_cls_labels > 0).float()
        postives_normal = torch.clamp(postives.sum(1,keepdim=True),min=1.0)
        dir_weigts = postives/postives_normal
        if self.forward_ret_dict['dir_preds'] is not None:
            dir_target = self.get_dir_target(cfg.MODEL.RPN.RPN_HEAD.ARGS["dir_offset"],
                                             cfg.MODEL.RPN.RPN_HEAD.ARGS["num_direction_bins"])
        dir_preds = self.forward_ret_dict['dir_preds'].reshape(batch_size,-1,cfg.MODEL.RPN.RPN_HEAD.ARGS["num_direction_bins"])
        dir_loss = self.dir_loss_layer(dir_preds,dir_target,dir_weigts)
        dir_loss = dir_loss.sum()/batch_size
        dir_loss = dir_loss * cfg.MODEL.RPN.RPN_HEAD.LOSSES["dir_loss_weight"]
        tb_dict = {"rpn_dir_loss":dir_loss}
        return dir_loss,tb_dict

    def get_dir_target(self,dir_offset,num_bins,one_hot=True):
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        batch_size = box_reg_targets.shape[0]
        batch_anchors = torch.cat([anchor for anchor in self.anchors],dim=-3)
        batch_anchors = batch_anchors.reshape(1,-1,self.box_code_size).repeat(batch_size,1,1)
        rot_angle = box_reg_targets[...,6]+batch_anchors[...,6]

        dir_cls_target = common_utils.limit_period(rot_angle-dir_offset,0,2*np.pi)
        dir_cls_target = torch.floor(dir_cls_target / (2*np.pi/num_bins)).long()
        dir_cls_target_np = dir_cls_target.cpu().numpy()

        dir_cls_target = torch.clamp(dir_cls_target, min=0, max=num_bins - 1)
        if one_hot:
            one_hot_dir_targets = torch.zeros(*dir_cls_target.shape,num_bins,
                                              dtype=dir_cls_target.dtype,
                                              device=dir_cls_target.device)
            one_hot_dir_targets.scatter_(-1,dir_cls_target.unsqueeze(dim=-1),1.0)
        return one_hot_dir_targets if one_hot else dir_cls_target



    def get_loss(self):
        loss_cls,tb_dict = self.get_cls_loss()
        loss_reg,tb_dict_box = self.get_reg_loss()
        loss_dir,tb_dict_dir = self.get_dir_loss()
        tb_dict.update(tb_dict_box)
        tb_dict.update(tb_dict_dir)
        loss_rpn = loss_reg+loss_cls + loss_dir
        tb_dict.update({"rpn_loss":loss_rpn.item()})
        return loss_rpn,tb_dict

    def predict_box(self):
        rpn_cls_preds = self.forward_ret_dict["cls_preds"]
        batch_size = rpn_cls_preds.shape[0]
        rpn_box_preds_src = self.forward_ret_dict["box_preds"].reshape(batch_size,-1,self.box_code_size)
        rpn_cls_preds = rpn_cls_preds.reshape(batch_size,-1,self.num_class)
        #角度处理
        num_dir_bins = cfg.MODEL.RPN.RPN_HEAD.ARGS["num_direction_bins"]
        dir_offset = cfg.MODEL.RPN.RPN_HEAD.ARGS["dir_offset"]
        dir_limit_offset = cfg.MODEL.RPN.RPN_HEAD.ARGS["dir_limit_offset"]
        dir_period = 2*np.pi/num_dir_bins
        dir_preds = self.forward_ret_dict["dir_preds"].permute(0,2,3,1).reshape(batch_size,-1,num_dir_bins)
        dir_cls_preds = torch.argmax(dir_preds,dim=-1)
        #box解码
        batch_anchors = torch.cat([anchor for anchor in self.anchors],dim=-3).reshape(1,-1,self.box_code_size).repeat(batch_size,1,1)
        rpn_box_preds = self.box_coder.decode_torch(rpn_box_preds_src,batch_anchors)
        rot_angle_preds = common_utils.limit_period(rpn_box_preds[...,6]-dir_offset,
                                                    offset=dir_limit_offset,period=dir_period)

        rot_angle_preds_final = rot_angle_preds+dir_offset+dir_period*dir_cls_preds.to(rpn_box_preds.dtype)
        # for_test = rot_angle_preds_final>2*np.pi
        # for_test = for_test.sum()
        rpn_box_preds[...,6] = rot_angle_preds_final % (np.pi*2)

        return rpn_cls_preds,rpn_box_preds




class RPNV2(AnchorHead):
    def __init__(self,num_class, args, anchor_target_cfg, grid_size, **kwargs):
        super().__init__(grid_size=grid_size, anchor_target_cfg=anchor_target_cfg)
        self._use_direction_classifier = args['use_direction_classifier']
        self._concat_input = args['concat_input']

        assert len(args['layer_strides']) == len(args['layer_nums'])
        assert len(args['num_filters']) == len(args['layer_nums'])
        assert len(args['num_upsample_filters']) == len(args['layer_nums'])
        self.forward_ret_dict = {}
        if args['use_norm']:
            BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
            Conv2d = partial(nn.Conv2d, bias=False)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        else:
            BatchNorm2d = Empty
            Conv2d = partial(nn.Conv2d, bias=True)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=True)

        in_filters = [args['num_input_features'], *args['num_filters'][:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(args['layer_nums']):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(in_filters[i], args['num_filters'][i], 3, stride=args['layer_strides'][i]),
                BatchNorm2d(args['num_filters'][i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                block.add(BatchNorm2d(args['num_filters'][i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    args['num_filters'][i], args['num_upsample_filters'][i], args['upsample_strides'][i],
                    stride=args['upsample_strides'][i]
                ),
                BatchNorm2d(args['num_upsample_filters'][i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)

        c_in = sum(args['num_upsample_filters'])
        if self._concat_input:
            c_in += args['num_input_features']

        if len(args['upsample_strides']) > len(args['num_filters']):
            deblock = Sequential(
                ConvTranspose2d(c_in, c_in, args['upsample_strides'][-1], stride=args['upsample_strides'][-1]),
                BatchNorm2d(c_in),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        if args['encode_background_as_zeros']:
            num_cls = self.num_anchors_per_location * num_class
        else:
            num_cls = self.num_anchors_per_location * (num_class + 1)
        self.conv_cls = nn.Conv2d(c_in, num_cls, 1)
        reg_channels = self.num_anchors_per_location * self.box_code_size
        self.conv_box = nn.Conv2d(c_in, reg_channels, 1)


        if args['use_direction_classifier']:
            self.conv_dir_cls = nn.Conv2d(c_in, self.num_anchors_per_location * args['num_direction_bins'], 1)
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0,std=0.001)


    def forward(self, x_in, bev=None, **kwargs):
        ups = []
        x = x_in
        ret_dict = {}
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(x_in.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x

            ups.append(self.deblocks[i](x))

        if self._concat_input:
            ups.append(x_in)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        if len(self.deblocks)>len(self.blocks):
            x = self.deblocks[-1](x)
        ret_dict['spatial_features_last'] = x

        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict.update({
            'box_preds': box_preds,
            'cls_preds': cls_preds,
        })
        if cfg.MODEL.RPN.RPN_HEAD.ARGS["use_direction_classifier"]:
            dir_pred = self.conv_dir_cls(x)
            ret_dict.update({"dir_preds":dir_pred.contiguous()})
        self.forward_ret_dict = ret_dict


        if self.training:
            targets_dict = self.get_assigner_target(
                gt_boxes=kwargs['gt_boxes'],
            )
            # 验证target
            # batch_size = targets_dict["box_reg_targets"].shape[0]
            # anchors = torch.cat([anchor for anchor in self.anchors], dim=-2)
            # anchors = anchors.reshape(1, -1, 7).repeat(batch_size, 1, 1)
            # box_reg_targets = targets_dict["box_reg_targets"]
            # box_reg_targets_np = box_reg_targets.cpu().numpy()
            # box_reg_targets_np = box_reg_targets_np[box_reg_targets_np!=0].reshape(batch_size,-1,7)
            # box_gt_recover = self.box_coder.decode_torch(box_reg_targets, anchors)
            # import pvdet.dataset.iou3d_nms.iou3d_nms_utils as iou3d_nms_utils
            # iou3d = iou3d_nms_utils.boxes_iou3d_gpu(box_gt_recover[0],kwargs["gt_boxes"][0][...,:7])
            # idx = (iou3d > 0.98).int()
            # idx = torch.nonzero(idx)
            # iou3d = iou3d[idx[:,0],idx[:,1]]
            # box_gt_recover = box_gt_recover[0][idx[:,0],:]
            ret_dict.update(targets_dict)
            self.forward_ret_dict.update(ret_dict)
            #loss,tb_dict = self.get_loss()

        rpn_cls_preds,rpn_box_preds = self.predict_box()
        ret_dict.update({"rpn_box_preds":rpn_box_preds,
                             "rpn_cls_preds":rpn_cls_preds})
        self.forward_ret_dict.update(ret_dict)
        return ret_dict

if __name__ == '__main__':
    # rot = torch.tensor(-np.pi/4)
    # rot = common_utils.limit_period(rot,0,2*np.pi)
    a = np.pi/3*2 - np.pi/4
    print("done")

#x:[2,128,200,176]->[2,256,100,88]
#ups[2,256,200,176]->[2,256,200,176]









