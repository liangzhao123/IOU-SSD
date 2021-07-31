import torch.nn as nn
import torch
from pvdet.ops.iou3d_nms import iou3d_nms_utils
from pvdet.tools.utils.box_coder_utils import ResidualCoder_v1
from pvdet.model.model_utils.generate_stage_two_targets import GenerateStageTwoTargets
from new_train.utils import loss_utils
from new_train.utils import common_utils
from pvdet.model.pointnet2.pointnet2_stack.pointnet2_modules import StackSAModuleMSG
import torch.nn.functional as F


def get_roi_from_rpn(box_score,box,nms_cfg):
    """
    box_score:(num of anchor,)
    box:(num of anchor, 7)
    nms_cfg:nms的配置文件
    """

    selected = []
    if box.shape[0]>0:
        rank_score,indices = torch.topk(box_score,k=min(nms_cfg.NMS_PRE_MAXSIZE,box.shape[0]))
        for_nms_box = box[indices]
        nms_indices, _ = iou3d_nms_utils.nms_gpu(for_nms_box,rank_score,nms_cfg.NMS_THRESH,**nms_cfg)
        selected = indices[nms_indices[:nms_cfg.NMS_POST_MAXSIZE]]

    return selected.view(-1),box_score[selected]

class RCNNbase(nn.Module):
    def __init__(self,model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.get_roi_from_rpn =get_roi_from_rpn
        self.get_proposal_targets = GenerateStageTwoTargets(model_cfg.TARGET_CONFIG)
        self.code_func = ResidualCoder_v1()
        self.build_loss()

    def build_loss(self):
        self.cls_loss_func= loss_utils.WeightedCrossEntropyLoss_v1()
        self.reg_loss_func = loss_utils.WeightedSmoothL1Loss_v1(
            code_weights=self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS["code_weights"])
        self.corner_loss_func = loss_utils.get_corner_loss_lidar_v1

    def proposal_layer(self,batch_dict,nms_cfg):
        batch_size = batch_dict['batch_size']
        rpn_box_preds = batch_dict['rpn_box_preds']
        rpn_cls_preds = batch_dict['rpn_cls_preds']
        roi = rpn_box_preds.new_zeros(batch_size,nms_cfg.NMS_POST_MAXSIZE,rpn_box_preds.shape[-1])
        roi_score = rpn_box_preds.new_zeros(batch_size,nms_cfg.NMS_POST_MAXSIZE)
        roi_labels = rpn_box_preds.new_zeros(batch_size,nms_cfg.NMS_POST_MAXSIZE,dtype=torch.long)
        for batch_index in range(batch_size):
            cur_box = rpn_box_preds[batch_index]
            cur_cls = rpn_cls_preds[batch_index]
            cur_roi_score,cur_roi_labels = torch.max(cur_cls,dim=1)
            selected,selected_score = self.get_roi_from_rpn(cur_roi_score,cur_box,nms_cfg)
            roi[batch_index,:len(selected),:] = cur_box[selected]
            roi_score[batch_index,:len(selected)] = cur_roi_score[selected]
            roi_labels[batch_index,:len(selected)] = cur_roi_labels[selected]

        batch_dict["rois"] = roi
        batch_dict["roi_labels"] = roi_labels + 1
        batch_dict["roi_scores"] = roi_score
        return batch_dict

    def assign_targets(self,batch_dict):
        rois = batch_dict["rois"]
        gt_boxes = batch_dict["gt_boxes"][...,:7]
        gt_class = batch_dict["gt_boxes"][...,7]
        with torch.no_grad():
            target_dict = self.get_proposal_targets.forward(batch_dict)
        return target_dict

    def get_reg_loss(self):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        reg_preds = self.forward_ret_dict["rcnn_reg"]
        reg_targets = self.forward_ret_dict['rcnn_reg_targets']
        reg_valid_mask = self.forward_ret_dict['fg_valid_reg_mask']
        batch_size = reg_targets.shape[0]
        reg_preds = reg_preds.view(batch_size, -1, reg_preds.shape[-1])

        fg_mask = (reg_valid_mask>0)
        try:
            fg_num = fg_mask.cpu().long().sum().item()

        except:
            print("fg_num: ",fg_mask)
            raise ValueError

        tb_dict = {"rcnn_fg_reg_num":fg_num}

        loss = 0.0
        if loss_cfgs.REG_LOSS == "smooth-l1":
            reg_loss = self.reg_loss_func(reg_preds,reg_targets)
            fg_flag = (fg_mask>0).float().unsqueeze(dim=-1)
            reg_loss = (reg_loss*fg_flag).sum()/torch.clamp(fg_flag.sum(),1.0)
            reg_loss = reg_loss*loss_cfgs.LOSS_WEIGHTS["rcnn_reg_weight"]
            tb_dict["rcnn_fg_reg_loss"] = reg_loss.item()
            loss +=reg_loss
            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_num>0:
                # reg_valid_mask[...,0:3] =1.0 for debug
                corner_targets = self.forward_ret_dict['rcnn_reg_corners_target']
                fg_reg_preds = reg_preds[fg_mask]
                rois = self.forward_ret_dict["rois"]
                fg_rois = rois[fg_mask]
                fg_rois_ry = fg_rois[...,6]
                fg_rois_xyz = fg_rois[...,0:3]
                fg_anchors = fg_rois.clone().detach()
                fg_anchors[...,0:3] = 0
                fg_box_preds_ct = self.code_func.decode_torch(fg_reg_preds,fg_anchors)
                fg_box_preds = common_utils.rotate_points_along_z(fg_box_preds_ct.unsqueeze(dim=1),fg_rois_ry).squeeze(dim=1)
                fg_box_preds[...,0:3] += fg_rois_xyz
                fg_corner_tagrets = corner_targets[fg_mask]
                corner_loss = self.corner_loss_func(fg_box_preds[...,0:7],fg_corner_tagrets[...,0:7])
                corner_loss = corner_loss.mean()
                corner_loss = corner_loss*loss_cfgs.LOSS_WEIGHTS["rcnn_corner_weight"]
                loss += corner_loss
                tb_dict.update({"rcnn_corner_loss":corner_loss.item()})
        return loss,tb_dict

    def get_cls_loss(self):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        cls_preds = self.forward_ret_dict["rcnn_cls"]
        cls_targets = self.forward_ret_dict['rcnn_cls_targets'].view(-1)
        if loss_cfgs.CLS_LOSS =="BinaryCrossEntropy":
            cls_loss = F.binary_cross_entropy(torch.sigmoid(cls_preds).view(-1),cls_targets.float(),reduction="none")
        else:
            raise NotImplementedError
        cls_loss_mask = (cls_targets>=0).float()
        cls_loss = (cls_loss*cls_loss_mask).sum()/torch.clamp(cls_loss_mask.sum(),min=1.0)
        cls_loss = cls_loss*loss_cfgs.LOSS_WEIGHTS["rcnn_cls_weight"]
        tb_dict = {"rcnn_cls_loss":cls_loss}
        return cls_loss,tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_loss()
        reg_loss, tb_reg_dict = self.get_reg_loss()
        loss = cls_loss + reg_loss
        tb_dict.update(tb_reg_dict)
        return loss, tb_dict

    def generate_predict_box(self,batch_size,cls_preds,reg_preds,rois):
        cls_preds = cls_preds.view(batch_size,-1,cls_preds.shape[-1])
        reg_preds = reg_preds.view(batch_size,-1,self.code_func.code_size)
        rois_y = rois[...,6].view(-1)
        rois_xyz = rois[...,0:3]
        rois_temp = rois.clone().detach()
        rois_temp[...,0:3] = 0
        box_ct = self.code_func.decode_torch(reg_preds,rois_temp).view(-1,self.code_func.code_size)
        box_preds = common_utils.rotate_points_along_z(
            box_ct.unsqueeze(dim=1),rois_y).squeeze(dim=1).view(batch_size,-1,self.code_func.code_size)
        box_preds[...,0:3] += rois_xyz
        return cls_preds,box_preds

class RCNNnet(RCNNbase):
    def __init__(self,channel_in,num_class,model_cfg):
        super().__init__(model_cfg)
        self.num_class = num_class
        self.forward_ret_dict = {}
        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        for k in range(len(mlps)):
            mlps[k] = [channel_in] + mlps[k]

        self.roi_pool_layer = StackSAModuleMSG(
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method="max_pool"
        )
        #池化后首先进行->特征共享层
        pool_out_channel = sum([mlp[-1] for mlp in mlps])
        pre_channel = self.model_cfg.ROI_GRID_POOL.GRID_SIZE**3*pool_out_channel
        share_modules = []
        share_channels = self.model_cfg.SHARED_FC
        for k in range(len(share_channels)):
            share_modules.extend(
                [
                    nn.Conv1d(pre_channel,share_channels[k],kernel_size=1,bias=False),
                    nn.BatchNorm1d(share_channels[k]),
                    nn.ReLU(),
                ]
            )
            pre_channel = share_channels[k]
            if k != len(share_channels)-1 and self.model_cfg.DP_RATIO>0:
                share_modules.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.share_layer = nn.Sequential(*share_modules)

        #类别全连接层
        cls_layer = []
        pre_channel = self.model_cfg.SHARED_FC[-1]
        cls_channel = self.model_cfg.CLS_FC
        for k in range(len(cls_channel)):
            cls_layer.extend(
                [
                    nn.Conv1d(pre_channel,cls_channel[k],kernel_size=1,stride=1,bias=False),
                    nn.BatchNorm1d(cls_channel[k]),
                    nn.ReLU()
                ]
            )
            pre_channel = cls_channel[k]
            if k != len(cls_channel)-1:
                cls_layer.append(nn.Dropout(self.model_cfg.DP_RATIO))
        cls_layer.append(nn.Conv1d(pre_channel,self.num_class,kernel_size=1,stride=1,bias=True))
        self.cls_layer = nn.Sequential(*cls_layer)

        #box回归全连接层
        reg_channels = self.model_cfg.REG_FC
        pre_channel = self.model_cfg.SHARED_FC[-1]
        reg_layer = []
        for k in range(len(reg_channels)):
            reg_layer.extend([
                nn.Conv1d(pre_channel,reg_channels[k],kernel_size=1,stride=1,bias=False),
                nn.BatchNorm1d(reg_channels[k]),
                nn.ReLU()
            ])
            pre_channel = reg_channels[k]
            if k != len(reg_channels)-1 :
                reg_layer.append(nn.Dropout(self.model_cfg.DP_RATIO))
        reg_layer.append(nn.Conv1d(pre_channel,self.code_func.code_size*self.num_class,kernel_size=1,stride=1,bias=True))
        self.reg_layer = nn.Sequential(*reg_layer)

        self.init_weights()

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].weight, mean=0, std=0.001)

    def roi_grid_pool_layer(self,batch_dict):
        batch_size = batch_dict["rois"].shape[0]
        rois = batch_dict["rois"].reshape(-1,batch_dict["rois"].shape[-1])
        point_coords = batch_dict["point_coords"]
        point_features = batch_dict['point_features']
        point_features = point_features* batch_dict["point_cls_scores"].view(-1,1)

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE

        #生成rois的网格点将每个roi分成6*6*6的网格
        rcnn_batch_size = rois.shape[0]
        local_grid = rois.new_ones(grid_size,grid_size,grid_size)
        local_grid_id = torch.nonzero(local_grid).float()
        local_grid_id = local_grid_id.repeat(rcnn_batch_size,1,1)
        rois_size = rois[...,3:6].unsqueeze(dim=1).clone()

        gloal_grid_points = (local_grid_id+0.5)*(rois_size/grid_size) - rois_size/2
        rois_ry = rois[...,6].clone()
        gloal_grid_points = common_utils.rotate_points_along_z(
            gloal_grid_points.clone(),rois_ry.view(-1))
        roi_centers = rois[...,0:3].clone()
        gloal_grid_points += roi_centers.unsqueeze(dim=1)

        gloal_grid_points = gloal_grid_points.view(batch_size,-1,gloal_grid_points.shape[-1])
        xyz = point_coords[:,1:4]
        xyz_batch_count = xyz.new_zeros(batch_size,).int()
        for bn_idx in range(batch_size):
            xyz_batch_count[bn_idx] = (point_coords[:,0] == bn_idx).sum()
        new_xyz = gloal_grid_points.view(-1,3)
        new_xyz_bn_count = new_xyz.new_zeros(batch_size).fill_(gloal_grid_points.shape[1]).int()
        pooled_xyz,pooled_features = self.roi_pool_layer(
            xyz.contiguous(),
            xyz_batch_count,
            new_xyz,
            new_xyz_bn_count,
            features = point_features
        )
        return pooled_features

    def forward(self,batch_dict):
        batch_dict = self.proposal_layer(
            batch_dict,
            self.model_cfg.NMS_CONFIG["TRAIN"] if self.training else self.model_cfg.NMS_CONFIG["TEST"])
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict["rois"] = targets_dict["rois"]
            batch_dict["roi_labels"] = targets_dict["roi_labels"]
            batch_dict["roi_scores"] = targets_dict["roi_scores"]
            self.forward_ret_dict.update(targets_dict)
        pooled_features = self.roi_grid_pool_layer(batch_dict)
        roi_grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pooled_features = pooled_features.view(-1,roi_grid_size**3,pooled_features.shape[-1])
        pooled_features = pooled_features.permute(0,2,1).contiguous()#(rcnn_batch_size,C,6*6*6)
        batch_rcnn_size =pooled_features.shape[0]
        share_features = self.share_layer(pooled_features.view(batch_rcnn_size,-1,1))
        cls_preds = self.cls_layer(share_features).squeeze(dim=-1).contiguous()
        reg_preds = self.reg_layer(share_features).squeeze(dim=-1).contiguous()


        if not self.training :
            batch_cls_preds, batch_box_preds = self.generate_predict_box(
                batch_dict['batch_size'],cls_preds,reg_preds,batch_dict["rois"])
            batch_dict.update({"rcnn_cls_preds":batch_cls_preds,
                                "rcnn_box_preds":batch_box_preds})

        else:
            self.forward_ret_dict.update({"rcnn_cls": cls_preds,
                                          "rcnn_reg": reg_preds})
            # loss, tb_dict = self.get_loss()
        return batch_dict