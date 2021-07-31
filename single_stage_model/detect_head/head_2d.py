import torch
import numpy as np
import torch.nn as nn
from single_stage_model.detect_head.Anchor_utils import AnchorGenertor
from single_stage_model.detect_head.box_coder import ResidualCoder_v1
import single_stage_model.utils.common_utils as common_utils
import single_stage_model.utils.loss_utils as  loss_utils
from single_stage_model.detect_head.target_assigner import AxisAlignedTargetAssigner


# import pvdet.ops.iou3d_nms.iou3d_nms_utils as iou3d_nms_utils
import single_stage_model.iou3d_nms.iou3d_nms_utils as iou3d_nms_utils

# from pvdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu,boxes_iou_bev
from single_stage_model.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu,boxes_iou_bev

# import pvdet.dataset.utils.box_utils as box_utils
import single_stage_model.dataset.box_utils as box_utils


class DetectHead(nn.Module):
    def __init__(self,grid_size,config,class_names):
        super().__init__()

        anchors, self.num_anchors_per_location = self.get_anchor(grid_size,anchor_generator_config = config)
        self.anchors = anchors
        self.ret = {}
        self.match_height = config["target_config"]["match_height"]
        self.config = config
        self.box_coder_fun = ResidualCoder_v1(code_size=config["target_config"]["code_size"])
        self.norm_by_num_samples = config["target_config"]["norm_by_num_samples"]
        self.num_class = config["num_class"]
        self.using_backgroud_as_zero = self.config["using_backgroud_as_zero"]

        self.conv_dir_cls = nn.Conv2d(1024, sum(self.num_anchors_per_location) * config["dir_cls_bin"], 1, bias=True)
        self.conv_box = nn.Conv2d(1024, sum(self.num_anchors_per_location)*7, 1, bias=True)

        if self.using_backgroud_as_zero:
            self.conv_cls = nn.Conv2d(1024, sum(self.num_anchors_per_location)*(self.num_class+1), 1, bias=True)
        else:
            self.conv_cls = nn.Conv2d(1024, sum(self.num_anchors_per_location)*(self.num_class), 1, bias=True)


        self.class_names = class_names
        target_cfg = config["target_config"]
        self.target_assigner = AxisAlignedTargetAssigner(
            anchor_cfg=target_cfg,
            class_names=self.class_names,
            box_coder=self.box_coder_fun
        )
        self.use_multihead = False
        self.using_iou_branch = config["using_iou_branch"]
        if self.using_iou_branch:
            self.conv_iou_branch = nn.Sequential(nn.Conv2d(1024, 36,1,  bias=True))
        self.init_weights()
        self.build_loss_layer(config["loss_config"])

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0,std=0.001)
        if self.using_iou_branch:
            nn.init.normal_(self.conv_iou_branch[-1].weight, mean=0, std=0.001)
            nn.init.constant_(self.conv_iou_branch[-1].bias, -np.log((1 - pi) / pi))
    def build_loss_layer(self,losses_cfg):
        self.cls_loss_layer = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        self.reg_loss_layer = loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg['code_loss_weight'])
        self.dir_loss_layer = loss_utils.WeightedCrossEntropyLoss()
        if self.using_iou_branch:
            self.iou_residual_loss_layer = loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg["iou_loss_weight"])
            self.iou_bin_loss_layer = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
    def get_anchor(self, grid_size, anchor_generator_config):
        point_cloud_range = anchor_generator_config['point_cloud_range']
        anchor_generator = AnchorGenertor(point_cloud_range,
                                          anchor_generator_config["target_config"]["anchor_generator"])

        features_map_size = [grid_size[:2] // config["feature_map_stride"] for
                             config in anchor_generator_config["target_config"]["anchor_generator"]]
        anchors_list, num_anchors_per_location_list = anchor_generator.generator(features_map_size,torch_enable=True)
        return anchors_list, num_anchors_per_location_list

    def get_assigner_target(self, gt_boxes):
        """
                :param gt_boxes: (B, N, 8)
                :return:
                """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes, self.use_multihead
        )

        return targets_dict



    def target_assigner_v0(self,batch_dict):
        batch_size = batch_dict["batch_size"]
        gt_boxes = batch_dict["gt_boxes"]
        anchors = torch.cat(self.anchors,dim=3)

        anchor_masks = batch_dict.get("anchor_masks",None)
        if anchor_masks is not None:
            anchor_masks = torch.from_numpy(anchor_masks[0]).cuda()

        reg_target_batch = gt_boxes.new_zeros(batch_size, *anchors.shape[1:]).view(batch_size,-1,len(self.anchors),self.box_coder_fun.code_size)
        labels_batch = torch.ones(batch_size,*reg_target_batch.shape[1:3],dtype=torch.float32,device=gt_boxes.device)*-1
        weight_batch = gt_boxes.new_zeros(batch_size,*reg_target_batch.shape[1:3])
        for bs_idx in range(batch_size):
            cur_gt_boxes = gt_boxes[bs_idx]
            cnt = len(cur_gt_boxes)-1
            while cnt>0 and cur_gt_boxes[cnt].sum()==0:
                cnt -=1
            cur_gt_boxes = cur_gt_boxes[:cnt+1]

            reg_target_single_list =[]
            labels_single_list = []
            weight_single_list = []
            for class_idx in range(1,len(self.anchors)+1):

                cur_anchors_src = self.anchors[class_idx - 1]
                cur_anchors_src = cur_anchors_src.view(-1, (cur_anchors_src.shape[-1]))
                num_anchors = cur_anchors_src.shape[0]

                if anchor_masks is not None:
                    cur_anchor_mask = anchor_masks[class_idx-1]
                    original_idx = torch.nonzero(cur_anchor_mask.float().view(-1) >0).view(-1)
                    cur_anchors = cur_anchors_src[cur_anchor_mask]
                    num_valid_anchor = original_idx.shape[0]
                else:
                    cur_anchors = cur_anchors_src
                    num_valid_anchor = num_anchors

                matched_threshold =  self.config["anchor_generator"][class_idx - 1]["matched_threshold"]
                unmatched_threshold = self.config["anchor_generator"][class_idx - 1]["unmatched_threshold"]

                gt_idx = torch.ones((num_anchors,),dtype=torch.int32,device=cur_gt_boxes.device)* -1
                labels = torch.ones((num_anchors,),dtype=torch.float32,device=cur_gt_boxes.device)* -1
                reg_target_single_class = cur_anchors.new_zeros((num_anchors,self.box_coder_fun.code_size))
                reg_weight_single_class = torch.zeros((num_anchors,),dtype=torch.float32,device=cur_gt_boxes.device)


                class_mask = (cur_gt_boxes[:,7]== class_idx)
                cur_same_gt_boxes = cur_gt_boxes[class_mask]
                num_gt = cur_same_gt_boxes.shape[0]

                if num_gt>0 and num_anchors>0 and num_valid_anchor>0:

                    if self.match_height:
                        iou = boxes_iou3d_gpu(cur_anchors,cur_same_gt_boxes[:,:7])
                    else:
                        iou = box_utils.boxes3d_nearest_bev_iou(cur_anchors,cur_same_gt_boxes[:,:7])
                    #anchor corresponded gt
                    gt_max_indx = torch.argmax(iou,dim=1)
                    anchors_gt_max = iou[torch.arange(num_valid_anchor),gt_max_indx] #[70400]

                    #gt corresponded anchor
                    anchors_max_indx = torch.argmax(iou,dim=0)
                    anchors_max_gt = iou[anchors_max_indx,torch.arange(num_gt)] #[num_gt]
                    empty_indx = anchors_max_gt==0 #invalid anchor
                    anchors_max_gt[empty_indx] = -1

                    optimal_match = torch.nonzero(iou == anchors_max_gt)
                    optimal_anchor_idx = optimal_match[:,0]
                    gt_force_indx = gt_max_indx[optimal_anchor_idx]

                    if anchor_masks is not None:
                        optimal_anchor_idx =  original_idx[optimal_anchor_idx]

                    gt_idx[optimal_anchor_idx] = gt_force_indx.int()
                    valid_force_labels = cur_same_gt_boxes[gt_force_indx][:,7]
                    labels[optimal_anchor_idx] = valid_force_labels

                    pos_idx = torch.nonzero(anchors_gt_max> matched_threshold).view(-1)
                    gt_idx_over_thresh = gt_max_indx[pos_idx]

                    if anchor_masks is not None:
                        pos_idx = original_idx[pos_idx]
                    gt_idx[pos_idx] = gt_idx_over_thresh.int()
                    valid_label = cur_same_gt_boxes[gt_idx_over_thresh][:,7]
                    labels[pos_idx] = valid_label

                    bg_idx = torch.nonzero(anchors_gt_max<unmatched_threshold).view(-1)
                    if anchor_masks is not None:
                        bg_idx = original_idx[bg_idx]

                else:
                    bg_idx = torch.arange(num_anchors)


                fg_idx = torch.nonzero(labels>0).view(-1)
                fg_idx_ = torch.nonzero(gt_idx>=0).view(-1)#:there is some index equal zero
                assert fg_idx.shape[0] == fg_idx_.shape[0]

                if cur_same_gt_boxes.shape[0]>0 and cur_anchors.shape[0]>0:
                    assert len(fg_idx)>0
                    fg_box_idx = gt_idx[fg_idx].long()
                    fg_gt_boxes = cur_same_gt_boxes[fg_box_idx]

                    fg_anchors = cur_anchors_src[fg_idx]
                    reg_target_same = self.box_coder_fun.encode_torch(fg_gt_boxes[:,:7],fg_anchors)
                    reg_target_single_class[fg_idx,:] = reg_target_same

                if len(cur_same_gt_boxes)==0 or cur_anchors.shape[0]==0:
                    labels[:] = 0
                else:
                    labels[bg_idx] = 0
                    #re-enable optimal labels
                    labels[optimal_anchor_idx] = valid_force_labels
                    fg_idx_v = torch.nonzero(labels>0).view(-1)
                    assert fg_idx.shape[0] == fg_idx_v.shape[0]

                if self.norm_by_num_samples:
                    num_pos_samples = len(fg_idx)
                    num_pos_samples = num_pos_samples if num_pos_samples>0 else 1.0
                    reg_weight = 1.0 /num_pos_samples
                    reg_weight_single_class[labels>0] = reg_weight
                else:
                    reg_weight_single_class[labels>0] = 1.0

                reg_target_single_list.append(reg_target_single_class)
                labels_single_list.append(labels)
                weight_single_list.append(reg_weight_single_class)

            reg_target_single = torch.stack(reg_target_single_list,dim=1)
            labels_single = torch.stack(labels_single_list,dim=-1)
            weight_single = torch.stack(weight_single_list,dim=-1)

            reg_target_batch[bs_idx] = reg_target_single
            labels_batch[bs_idx] = labels_single
            weight_batch[bs_idx] = weight_single

        ret = {"reg_target_batch":reg_target_batch.view(batch_size,-1,self.box_coder_fun.code_size), #[batch-size,70400,3,7]
               "labels_batch":labels_batch.view(batch_size,-1), #[batch-size,70400,3,]
               "weight_batch":weight_batch.view(batch_size,-1)} #[batch-size,70400,3,]

        return ret

    def get_dir_labels(self,one_hot=True):
        dir_offset = self.config["dir_offset"]
        reg_targets = self.ret["box_reg_targets"]
        num_bin = self.config["dir_cls_bin"]
        labels = self.ret["box_cls_labels"]
        pos_idx = labels>0
        batch_size = labels.shape[0]
        anchor_batch = torch.cat([anchor for anchor in self.anchors], dim=-3)
        anchor_batch = anchor_batch.reshape(1, -1, self.box_coder_fun.code_size).repeat(batch_size, 1, 1)

        rot_y = reg_targets[...,6]+anchor_batch[...,6]

        offset_rot = common_utils.limit_period(rot_y - dir_offset, 0, 2 * np.pi)
        dir_cls_target = torch.floor(offset_rot/(2*np.pi/num_bin))
        dir_cls_target = torch.clamp(dir_cls_target,min=0,max=num_bin-1).long()
        if one_hot:
            one_hot_dir_target = dir_cls_target.new_zeros(*dir_cls_target.shape,num_bin)
            one_hot_dir_target.scatter_(-1,dir_cls_target.unsqueeze(-1),1.0)
            dir_cls_target = one_hot_dir_target if one_hot else dir_cls_target

        return dir_cls_target



    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    def test_decode_encode(self,batch_dict):
        gt_boxes = batch_dict["gt_boxes"]
        anchors = torch.cat(self.anchors, dim=3)
        pass

    def get_reg_loss(self):
        tb_dict = {}
        box_preds_ = self.ret["box_preds_"]

        batch_size = box_preds_.shape[0]

        bbox_targets = self.ret['box_reg_targets']
        labels = self.ret['box_cls_labels']
        # weight = self.ret["weight_batch"]

        pos = (labels > 0).float()
        pos_normal = torch.sum(pos, dim=1, keepdim=True)
        pos_normal = torch.clamp(pos_normal, min=1.0)
        pos /= pos_normal
        weight = pos

        # sin(a-b) = sina*cosb-cosa*sinb
        box_preds_with_sin, bbox_targets_with_sin= self.add_sin_difference(box_preds_,bbox_targets)
        loss_reg = self.reg_loss_layer(box_preds_with_sin, bbox_targets_with_sin, weight)
        loss_reg = torch.sum(loss_reg) / batch_size
        loss_reg = loss_reg * self.config["loss_config"]["reg_loss_weight"]
        tb_dict.update({"rpn_box_loss":loss_reg})

        dir_cls_target = self.get_dir_labels()
        dir_preds = self.ret["dir_preds"]
        dir_loss = self.dir_loss_layer(dir_preds,dir_cls_target,weight)
        dir_loss = torch.sum(dir_loss)/batch_size
        dir_loss = self.config["loss_config"]["dir_loss_weight"] * dir_loss

        tb_dict.update({"rpn_dir_loss": dir_loss})

        return loss_reg+dir_loss,tb_dict

    def get_cls_loss(self,one_hot=True):
        tb_dict = {}
        labels = self.ret['box_cls_labels']
        cared = labels>=0
        cls_target = labels * cared.type_as(labels)

        cls_preds = self.ret["cls_preds"]
        pos = (labels>0).float()
        neg = (labels==0).float()
        cls_weight = pos+neg
        batch_size = labels.shape[0]

        pos_normal = torch.sum(pos,dim=1,keepdim=True)
        pos_normal = torch.clamp(pos_normal,min=1.0)
        cls_weight /= pos_normal

        if one_hot:
            one_hot_target = cls_target.new_zeros(*cls_target.shape,self.num_class+1)
            one_hot_target.scatter_(-1,cls_target.unsqueeze(dim=-1).long(),1)
        if self.using_backgroud_as_zero:
            one_hot_target = one_hot_target
            assert one_hot_target.shape[2] == 4

        else:
            one_hot_target = one_hot_target[:, :, 1:]
        cls_loss = self.cls_loss_layer(cls_preds,one_hot_target,weights = cls_weight)
        cls_loss = torch.sum(cls_loss)/batch_size
        cls_loss = cls_loss * self.config["loss_config"]["cls_weight"]

        tb_dict.update({"rpn_cls_loss":cls_loss})
        return cls_loss,tb_dict

    def predict_box(self):
        rpn_cls_preds = self.ret["cls_preds"]
        batch_size = rpn_cls_preds.shape[0]
        rpn_box_preds_src = self.ret["box_preds_"]

        #角度处理
        num_dir_bins = self.config["dir_cls_bin"]
        dir_offset = self.config["dir_offset"]
        dir_limit_offset = self.config["dir_limit_offset"]
        dir_period = 2*np.pi/num_dir_bins
        dir_preds = self.ret["dir_preds"]
        dir_cls_preds = torch.argmax(dir_preds,dim=-1)
        #box解码
        batch_anchors = torch.cat([anchor for anchor in self.anchors], dim=-2)
        batch_anchors = batch_anchors.reshape(1, -1, 7).repeat(batch_size, 1, 1)
        # batch_anchors = torch.cat([anchor.view(-1,7) for anchor in self.anchors],dim=-1).\
        #     reshape(1,-1,self.box_coder_fun.code_size).repeat(batch_size,1,1)
        rpn_box_preds = self.box_coder_fun.decode_torch(rpn_box_preds_src,batch_anchors)
        rot_angle_preds = common_utils.limit_period(rpn_box_preds[...,6]-dir_offset,
                                                    offset=dir_limit_offset,period=dir_period)

        rot_angle_preds_final = rot_angle_preds+dir_offset+dir_period*dir_cls_preds.to(rpn_box_preds.dtype)
        # for_test = rot_angle_preds_final>2*np.pi
        # for_test = for_test.sum()
        rpn_box_preds[...,6] = rot_angle_preds_final % (np.pi*2)

        return rpn_cls_preds,rpn_box_preds

    def get_iou_labels(self,batch_dict):
        batch_size = batch_dict["batch_size"]
        gt_boxes = batch_dict["gt_boxes"]
        box_preds = self.ret["rpn_box_preds"]
        iou_labels_shapes = list(box_preds.shape[:2]) + [self.config["iou_bin_num"]]
        batch_iou_labels_src = torch.zeros(size=list(box_preds.shape[:2]),device=box_preds.device,dtype=torch.float32)
        batch_iou_labels_bin = torch.zeros(size=iou_labels_shapes,device=box_preds.device,dtype=torch.float32)
        batch_iou_labels_residual = torch.zeros(size=list(box_preds.shape[:2]),device=box_preds.device,dtype=torch.float32)
        for i in range(batch_size):
            cur_gt_boxes = gt_boxes[i]
            cur_pred_boxes = box_preds[i]
            cnt = len(cur_gt_boxes)- 1
            while cnt > 0 and cur_gt_boxes[cnt].sum() == 0:
                cnt -= 1
            cur_gt_boxes = cur_gt_boxes[:cnt + 1]
            if len(cur_gt_boxes) != 0 and len(cur_pred_boxes) !=0:
                cur_iou_labels_bin = batch_iou_labels_bin[i]

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_pred_boxes,cur_gt_boxes[...,:7])
                iou3d_max,_ = torch.max(iou3d,dim=1)
                iou3d_max_bin = torch.floor(iou3d_max/0.2).int()
                iou3d_max_bin = torch.clamp(iou3d_max_bin,max=4.0)
                # iou3d_max_bin_np = iou3d_max_bin.cpu().detach().numpy()
                # iou3d_max_bin_np = iou3d_max_bin_np[iou3d_max_bin_np!=0]
                cur_iou_labels_bin.scatter_(-1,iou3d_max_bin.unsqueeze(dim=-1).long(),1)

                cur_iou_labels_residual = iou3d_max - iou3d_max_bin * 0.2

                #for test the iou_bin target
                # pos_id = cur_iou_labels_bin.argmax(dim=-1) * 0.2
                # recover = cur_iou_labels_residual + pos_id
                # recover_np = recover.cpu().detach().numpy()
                # iou3d_max_np = recover.cpu().detach().numpy()
                # if recover_np.any() == iou3d_max_np.any():
                #     print("right !")
                # recover_np = recover_np[recover_np != 0]
                # iou3d_max_np = iou3d_max_np[iou3d_max_np != 0]
                # mask = (recover_np != iou3d_max_np)
                # recover_np = recover_np[mask]
                # iou3d_max_np = iou3d_max_np[mask]

                batch_iou_labels_bin[i,...] = cur_iou_labels_bin
                batch_iou_labels_residual[i,...] = cur_iou_labels_residual
                batch_iou_labels_src[i,...] = iou3d_max
                # iou3d_max_np = iou3d_max.cpu().detach().numpy()
                # batch_iou_labels_np = batch_iou_labels.cpu().detach().numpy()
                # iou3d_max_np = iou3d_max_np[iou3d_max_np>0.5]
                # batch_iou_labels_np = batch_iou_labels_np[batch_iou_labels_np>0.5]

        self.ret.update({"iou_labels_bin":batch_iou_labels_bin,
                         "iou_labels_residual":batch_iou_labels_residual,
                         "iou_labels_src":batch_iou_labels_src})




    def get_iou_branch_loss(self):
        iou_labels_bin = self.ret["iou_labels_bin"]
        iou_labels_residual = self.ret["iou_labels_residual"]
        iou_labels_src = self.ret["iou_labels_src"]

        iou_preds = self.ret["iou_preds"]
        iou_preds_bin = iou_preds[...,:5]
        iou_preds_residual = iou_preds[...,5:]

        batch_size = iou_preds.shape[0]


        pos = (iou_labels_src>0).float()
        pos_normal= torch.sum(pos)
        pos_normal = torch.clamp(pos_normal,min=1.0)
        neg = (iou_labels_src==0).float()
        weights_bin = pos+neg
        weights_bin /= pos_normal

        weights_residual = pos
        weights_residual /= pos_normal
        # weight_np = iou_weight.cpu().detach().numpy()
        # weight_np = weight_np[weight_np>0]
        # neg_np = neg[neg > 0]
        # pos_np = pos[pos > 0]
        iou_residual_loss = self.iou_residual_loss_layer(iou_preds_residual, iou_labels_residual.unsqueeze(dim=-1),weights_residual)
        iou_residual_loss = torch.sum(iou_residual_loss)/batch_size
        iou_residual_loss = iou_residual_loss*self.config["loss_config"]["iou_loss_residual_weight"]
        iou_bin_loss = self.iou_bin_loss_layer(iou_preds_bin,iou_labels_bin,weights_bin)
        iou_bin_loss = torch.sum(iou_bin_loss) / batch_size
        iou_bin_loss = iou_bin_loss * self.config["loss_config"]["iou_loss_bin_weight"]
        iou_loss = iou_residual_loss+iou_bin_loss
        tb_dict = {}
        tb_dict.update({"iou_loss": iou_loss})
        return iou_loss,tb_dict


    def get_loss(self):

        reg_loss,tb_dict = self.get_reg_loss()
        cls_loss, tb_dict_ = self.get_cls_loss()
        if self.using_iou_branch:
            iou_loss,tb_dict__= self.get_iou_branch_loss()
            tb_dict.update(tb_dict__)
            loss = cls_loss + reg_loss + iou_loss
        else:
            loss = cls_loss + reg_loss
        tb_dict.update(tb_dict_)

        return loss,tb_dict



    def forward(self,batch_dict):

        x_in = batch_dict["conv2d_features"]
        batch_size = batch_dict["batch_size"]
        box_preds_ = self.conv_box(x_in)
        cls_preds = self.conv_cls(x_in)
        dir_preds = self.conv_dir_cls(x_in)
        if self.using_iou_branch:
            # x_iou_in = batch_dict["conv2d_last_features"]
            iou_preds = self.conv_iou_branch(x_in)
            iou_preds = iou_preds.permute(0, 2, 3, 1).contiguous()
        target_dict = {}
        if self.training:
            target_dict = self.get_assigner_target(batch_dict["gt_boxes"])
            self.ret.update(target_dict)


        box_preds_ = box_preds_.permute(0,2,3,1).contiguous()
        cls_preds = cls_preds.permute(0,2,3,1).contiguous()
        dir_preds = dir_preds.permute(0,2,3,1).contiguous()


        if self.using_iou_branch:
            self.ret.update({
                "box_preds_": box_preds_.view(batch_size, -1, self.box_coder_fun.code_size),
                "cls_preds": cls_preds.view(batch_size, -1, len(self.anchors)),
                "dir_preds": dir_preds.view(batch_size, -1, self.config["dir_cls_bin"]),  # [batch_size,211200,2]
                "iou_preds": iou_preds.view(batch_size,-1,self.config["iou_bin_num"])
            })
        else:
            self.ret.update({
                "box_preds_":box_preds_.view(batch_size,-1,self.box_coder_fun.code_size),
                "cls_preds":cls_preds.view(batch_size,-1,len(self.anchors)),
                "dir_preds":dir_preds.view(batch_size,-1,self.config["dir_cls_bin"]) #[batch_size,211200,2]
            })

        rpn_cls_preds, rpn_box_preds = self.predict_box()
        self.ret.update({"rpn_cls_preds":rpn_cls_preds,
                         "rpn_box_preds":rpn_box_preds})


        if self.training:
            # batch_size = target_dict['box_reg_targets'].shape[0]
            # anchors = torch.cat([anchor for anchor in self.anchors], dim=-2)
            # anchors = anchors.reshape(1, -1, 7).repeat(batch_size, 1, 1)
            # box_reg_targets = target_dict['box_reg_targets']
            # box_reg_targets_np = box_reg_targets.cpu().numpy()
            # box_reg_targets_np = box_reg_targets_np[box_reg_targets_np!=0].reshape(batch_size,-1,7)
            # box_gt_recover = self.box_coder_fun.decode_torch(box_reg_targets, anchors)
            #
            # iou3d = iou3d_nms_utils.boxes_iou3d_gpu(box_gt_recover[0],batch_dict["gt_boxes"][0][...,:7])
            # idx = (iou3d > 0.98).int()
            # idx = torch.nonzero(idx)
            # iou3d = iou3d[idx[:,0],idx[:,1]]
            # box_gt_recover = box_gt_recover[0][idx[:,0],:]
            if self.using_iou_branch:
                self.get_iou_labels(batch_dict)
            # loss = self.get_loss()
        batch_dict.update(self.ret)
        return batch_dict


if __name__ == '__main__':
    pass