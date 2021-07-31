import torch
import torch.nn as nn
import single_stage_model.utils.loss_utils as  loss_utils
from single_stage_model.iou3d_nms import iou3d_nms_utils
import numpy as np

class IouHead(nn.Module):
    def __init__(self,config,in_channel=10):
        super().__init__()
        self.ret = {}
        self.conv_share = nn.Sequential(nn.Conv1d(in_channel,32,3,padding=1,bias=False),
                                        nn.BatchNorm1d(32),
                                        nn.ReLU(),
                                        nn.Conv1d(32, 64,3,padding=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU()
                                       )
        self.conv_bin = nn.Conv1d(64, 5,1, bias=True)
        self.conv_residual = nn.Conv1d(64, 1,1, bias=True)

        self.config = config
        # self.init_weight()
        self.build_loss()
    def init_weight(self):
        pi = 0.01
        nn.init.constant_(self.conv_bin.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_residual.weight, mean=0,std=0.001)

    def build_loss(self):
        self.iou_residual_loss_layer = loss_utils.WeightedSmoothL1Loss(code_weights=self.config["iou_loss_weight"])
        if self.config["using_cross_entropy"]:
            self.iou_bin_loss_layer = loss_utils.WeightedCrossEntropyLoss()
        else:
            self.iou_bin_loss_layer = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)

    def get_iou_labels(self,batch_dict):
        batch_size = batch_dict["batch_size"]
        gt_boxes = batch_dict["gt_boxes"]
        #TODO
        box_preds = self.ret["box_preds_selected"]
        iou_labels_shapes = list(box_preds.shape[:2]) + [self.config["iou_bin_num"]]
        batch_iou_labels_src = torch.zeros(size=list(box_preds.shape[:2]),device=box_preds.device,dtype=torch.float32)
        batch_iou_labels_bin = torch.zeros(size=iou_labels_shapes,device=box_preds.device,dtype=torch.float32)
        batch_iou_labels_residual = torch.zeros(size=list(box_preds.shape[:2]),device=box_preds.device,dtype=torch.float32)
        # print("batch_size:",batch_size)
        # print("gt_boxes.shape",gt_boxes.shape)
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


    def get_loss(self):
        iou_labels_bin = self.ret["iou_labels_bin"]
        iou_labels_residual = self.ret["iou_labels_residual"]
        iou_labels_src = self.ret["iou_labels_src"]

        iou_preds_bin = self.ret["iou_preds_bin"]

        iou_preds_residual = self.ret["iou_preds_residual"]

        batch_size = iou_preds_bin.shape[0]

        pos = (iou_labels_src > 0).float()
        pos_normal = torch.sum(pos)
        pos_normal = torch.clamp(pos_normal, min=1.0)
        neg = (iou_labels_src == 0).float()
        weights_bin = pos+neg
        weights_bin /= pos_normal

        weights_residual = pos+neg
        weights_residual /= pos_normal
        # weight_np = iou_weight.cpu().detach().numpy()
        # weight_np = weight_np[weight_np>0]
        # neg_np = neg[neg > 0]
        # pos_np = pos[pos > 0]
        iou_residual_loss = self.iou_residual_loss_layer(iou_preds_residual, iou_labels_residual.unsqueeze(dim=-1),
                                                         weights_residual)

        iou_residual_loss = torch.sum(iou_residual_loss) / batch_size

        iou_residual_loss = iou_residual_loss * self.config["iou_loss_residual_weight"]
        if iou_residual_loss>100:
            raise ValueError
        iou_residual_loss = torch.clamp(iou_residual_loss, max=100)
        iou_bin_loss = self.iou_bin_loss_layer(iou_preds_bin, iou_labels_bin, weights_bin)

        iou_bin_loss = torch.sum(iou_bin_loss) / batch_size
        iou_bin_loss = iou_bin_loss * self.config["iou_loss_bin_weight"]
        if iou_bin_loss>100:
            raise ValueError
        iou_bin_loss = torch.clamp(iou_bin_loss, max=100)
        iou_loss = iou_residual_loss + iou_bin_loss
        tb_dict = {}
        tb_dict.update({"iou_bin_loss": iou_bin_loss,
                        "iou_residual_loss":iou_residual_loss})
        return iou_loss, tb_dict

    def proposal_layer(self,batch_dict):
        box_preds = batch_dict["rpn_box_preds"]
        cls_preds = batch_dict["rpn_cls_preds"]
        batch_size = batch_dict["batch_size"]
        code_size = box_preds.shape[-1]
        num_cls = cls_preds.shape[-1]
        box_preds_selected = box_preds.new_zeros((batch_size,self.config["selected_num"],code_size))
        cls_preds_selected = cls_preds.new_zeros(((batch_size,self.config["selected_num"],num_cls)))
        for bs_id in range(batch_size):
            cur_box_preds = box_preds[bs_id]
            cur_cls_preds = cls_preds[bs_id]
            cur_box_score, cur_cls_label = torch.max(cur_cls_preds, dim=1)
            cur_box_score = torch.sigmoid(cur_box_score)

            selected = []
            try:
                # selected_score, selected = torch.topk(cur_box_score, k=min(self.config["selected_num"], cur_box_preds.shape[0]))
                # print("cur_box_score.shape", cur_box_score.shape)
                # print("self.config[\"selected_num\"] , cur_box_preds.shape[0]",
                #       self.config["selected_num"], cur_box_preds.shape[0])
                selected_score, selected = torch.topk(cur_box_score,
                                                      k=self.config["selected_num"])
            except:
                selected_score, selected = torch.topk(cur_box_score,
                                                      k=self.config["selected_num"])
                # print("error self.config[\"selected_num\"] , cur_box_preds.shape[0]",
                #       self.config["selected_num"],cur_box_preds.shape[0])
                # print("cur_box_score.shape",cur_box_score.shape)
                raise ModuleNotFoundError
            cur_cls_preds = cur_cls_preds[selected]
            cur_box_preds = cur_box_preds[selected]
            box_preds_selected[bs_id,...] = cur_box_preds
            cls_preds_selected[bs_id,...] = cur_cls_preds

        self.ret.update({"box_preds_selected":box_preds_selected,
                         "cls_preds_selected":cls_preds_selected})
        return box_preds_selected,cls_preds_selected

    def forward(self,batch_dict):
        box_preds_selected,cls_preds_selected = self.proposal_layer(batch_dict)
        box_preds_selected = box_preds_selected.permute(0,2,1).contiguous()
        cls_preds_selected = cls_preds_selected.permute(0,2,1).contiguous()
        x_in = torch.cat([box_preds_selected,cls_preds_selected],dim=1)
        features_share = self.conv_share(x_in)
        iou_preds_bin = self.conv_bin(features_share)
        iou_preds_residual = self.conv_residual(features_share)

        iou_preds_bin = iou_preds_bin.permute(0,2,1).contiguous()
        iou_preds_residual = iou_preds_residual.permute(0,2,1).contiguous()
        self.ret.update({"iou_preds_bin":iou_preds_bin,
                         "iou_preds_residual":iou_preds_residual})
        batch_dict.update({"iou_preds_bin":iou_preds_bin,
                           "iou_preds_residual":iou_preds_residual,
                           "box_preds_selected":box_preds_selected.permute(0,2,1).contiguous(),
                           "cls_preds_selected":cls_preds_selected.permute(0,2,1).contiguous()})
        if self.training:
            self.get_iou_labels(batch_dict)
            # loss= self.get_loss()
        # print("done")
        return batch_dict
