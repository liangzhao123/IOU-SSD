import torch.nn as nn
import torch
import numpy as np
from pvdet.ops.iou3d_nms import iou3d_nms_utils
import pvdet.dataset.utils.common_utils as common_utils
from pvdet.tools.utils.box_coder_utils import ResidualCoder_v1

class GenerateStageTwoTargets(nn.Module):
    def __init__(self,target_cfg):
        super().__init__()
        self.target_cfg = target_cfg
        self.code_func = ResidualCoder_v1()

    def forward(self,batch_dict):
        batch_rois_src,batch_roi_labels_src,batch_rois_scores_src,\
        batch_gt_of_rois_src,batch_rois_max_iou3d_src = self.get_sampled_rois(batch_dict)
        #生成回归的mask
        fg_valid_reg_mask = (batch_rois_max_iou3d_src > self.target_cfg.REG_FG_THRESH).long()

        #生成iou为引导的roi, score作为rcnn阶段类别损失函数的target
        rcnn_cls_targets = batch_rois_max_iou3d_src.clone().detach()
        batch_size = batch_dict["batch_size"]

        fg_mask = rcnn_cls_targets>self.target_cfg.CLS_FG_THRESH
        bg_mask = rcnn_cls_targets<self.target_cfg.CLS_BG_THRESH
        interval_mask = (fg_mask == 0) & (bg_mask==0)
        rcnn_cls_targets[fg_mask] = 1.0
        rcnn_cls_targets[bg_mask] = 0.0
        rcnn_cls_targets[interval_mask] = (rcnn_cls_targets[interval_mask] - 0.25)*2
        rcnn_cls_targets = rcnn_cls_targets.float()

        #局部坐标变换
        batch_gt_of_rois = batch_gt_of_rois_src.clone().detach()
        batch_rois = batch_rois_src.clone().detach()
        roi_centers = batch_rois[...,0:3]
        roi_rot = batch_rois[...,6]
        batch_gt_of_rois[...,6] = batch_gt_of_rois[...,6] % (2 * np.pi)
        batch_gt_of_rois[...,6] = batch_gt_of_rois[...,6]-roi_rot
        batch_gt_of_rois[...,0:3] = batch_gt_of_rois[...,0:3] -roi_centers
        batch_gt_of_rois = common_utils.rotate_points_along_z(
                    batch_gt_of_rois.reshape(-1,1,batch_gt_of_rois.shape[-1]),-roi_rot.view(-1))
        #对gt_of_roi的角度进行变换,
        gt_of_roi_ry = batch_gt_of_rois[...,6] %(np.pi*2)#(0-2*pi)
        opposite_flag = (gt_of_roi_ry>np.pi*0.5) & (gt_of_roi_ry < np.pi*1.5)
        gt_of_roi_ry[opposite_flag] = (gt_of_roi_ry[opposite_flag] -np.pi) % (np.pi*2)#(0 ~ pi*0.5, 1.5*pi ~ 2*pi)
        flag = gt_of_roi_ry>np.pi
        gt_of_roi_ry[flag] = gt_of_roi_ry[flag] - np.pi*2 #(-pi*0.5,pi*0.5)
        gt_of_roi_ry = torch.clamp(gt_of_roi_ry,min=-np.pi/2,max=np.pi/2)
        batch_gt_of_rois[...,6] = gt_of_roi_ry
        batch_gt_of_rois = batch_gt_of_rois.reshape(batch_size,-1,self.code_func.code_size+1)

        #生成回归目标
        batch_rois[...,0:3] = 0
        batch_rois[...,6] = 0
        rcnn_reg_targets = self.code_func.encode_torch(batch_gt_of_rois[...,:7],batch_rois)

        #生成corner loss的targets
        rcnn_reg_corners_target = batch_gt_of_rois_src

        ret_dict = { "rois":batch_rois_src,
                     "roi_scores":batch_rois_scores_src,
                    "roi_labels":batch_roi_labels_src,
                    "gt_of_rois_src":batch_gt_of_rois_src,
                     "gt_of_rois":batch_gt_of_rois,
                    "roi_iou3d":batch_rois_max_iou3d_src,
                     "fg_valid_reg_mask":fg_valid_reg_mask,
                    "rcnn_reg_targets":rcnn_reg_targets,
                    "rcnn_reg_corners_target":rcnn_reg_corners_target,
                    "rcnn_cls_targets":rcnn_cls_targets}
        return ret_dict
    def get_sampled_rois(self,batch_dict):
        rois_src= batch_dict["rois"]
        roi_labels_src = batch_dict["roi_labels"]
        roi_scores_src = batch_dict['roi_scores']
        gt_boxes_src = batch_dict["gt_boxes"]
        batch_size = batch_dict["batch_size"]
        code_size = batch_dict["rois"].shape[-1]

        #创建采样后的rois,rois_score,roi_labels,gt_of_roi,roi_max_iou
        batch_rois = rois_src.new_zeros(batch_size,self.target_cfg.ROI_PER_IMAGE,code_size)
        batch_rois_max_iou3d = rois_src.new_zeros(batch_size,self.target_cfg.ROI_PER_IMAGE)
        gt_of_rois = gt_boxes_src.new_zeros(batch_size,self.target_cfg.ROI_PER_IMAGE,code_size+1)
        batch_roi_labels = roi_labels_src.new_zeros(batch_size,self.target_cfg.ROI_PER_IMAGE)
        batch_rois_scores = rois_src.new_zeros(batch_size,self.target_cfg.ROI_PER_IMAGE)

        for batch_index in range(batch_size):
            cur_gt = gt_boxes_src[batch_index]
            cur_rois = rois_src[batch_index]
            cur_roi_labels = roi_labels_src[batch_index]
            cur_roi_scores = roi_scores_src[batch_index]
            k = cur_gt.__len__() -1
            while k>0 and cur_gt[k].sum() ==0:
                k -=1
            cur_gt = cur_gt[:k+1]
            cur_gt = cur_gt.new_zeros(1,cur_gt.shape[1]) if len(cur_gt)==0 else cur_gt
            max_overlap,gt_assignment = self.get_iou_with_same_class(cur_rois,cur_roi_labels,cur_gt)
            sampled_index = self.sample_rois(max_overlap)
            sampled_gt_assign = gt_assignment[sampled_index]

            batch_rois[batch_index] = cur_rois[sampled_index]
            batch_roi_labels[batch_index] = cur_roi_labels[sampled_index]
            gt_of_rois[batch_index] = cur_gt[sampled_gt_assign]
            batch_rois_max_iou3d[batch_index] = max_overlap[sampled_index]
            batch_rois_scores[batch_index]  = cur_roi_scores[sampled_index]

        return batch_rois,batch_roi_labels,batch_rois_scores,gt_of_rois,batch_rois_max_iou3d

    def sample_rois(self,max_overlap):
        roi_per_image = self.target_cfg.ROI_PER_IMAGE
        fg_thresh = min(self.target_cfg.CLS_FG_THRESH,self.target_cfg.REG_FG_THRESH)
        fg_index = torch.nonzero( (max_overlap>= fg_thresh) ).view(-1)
        easy_bg_index = torch.nonzero(max_overlap<self.target_cfg.CLS_BG_THRESH_LO).view(-1)
        hard_bg_index = torch.nonzero((max_overlap<fg_thresh)&(max_overlap>=self.target_cfg.CLS_BG_THRESH_LO)).view(-1)
        num_fg = fg_index.numel()
        num_bg = easy_bg_index.numel() + hard_bg_index.numel()
        if num_fg>0 and num_bg>0:
            sample_fg_num = min(int(roi_per_image*self.target_cfg.FG_RATIO),fg_index.shape[0])
            sample_bg_num = roi_per_image - sample_fg_num
            sampled_fg_index = fg_index[torch.randint(0,fg_index.numel(),size=(sample_fg_num,)).long()]
            sampled_bg_index = self.sample_bg(easy_bg_index,hard_bg_index,sample_bg_num)
            sampled_index = torch.cat([sampled_fg_index,sampled_bg_index],dim=0)
        elif num_fg==0 and num_bg>0:
            sample_bg_num = roi_per_image
            sample_bg_index = self.sample_bg(easy_bg_index,hard_bg_index,sample_bg_num)
            sampled_index = sample_bg_index
        elif num_fg>0 and num_bg==0:
            sample_fg_num = roi_per_image
            sampled_fg_index = fg_index[torch.randint(0, fg_index.numel(), size=(sample_fg_num,)).long()]
            sampled_index = sampled_fg_index
        else:
            raise NotImplementedError
        return sampled_index

    def sample_bg(self,easy_bg,hard_bg,num):
        if easy_bg.numel()>0 and hard_bg.numel()>0:
            hard_num = min(int(num*self.target_cfg.HARD_BG_RATIO),hard_bg.numel())
            hard_bg_sampled = hard_bg[torch.randint(low=0,high=hard_bg.numel(),size=(hard_num,)).long()]
            easy_num = num-hard_num
            easy_bg_sampled = easy_bg[torch.randint(0,easy_bg.numel(),size=(easy_num,)).long()]
            bg_sampled = torch.cat([easy_bg_sampled,hard_bg_sampled],dim=0)
        elif easy_bg.numel()>0 and hard_bg.numel() ==0:
            easy_num = num
            easy_bg_sampled = easy_bg[torch.randint(0, easy_bg.numel(), size=(easy_num,)).long()]
            bg_sampled = easy_bg_sampled
        elif easy_bg.numel() ==0 and hard_bg.numel()>0:
            hard_num =num
            hard_bg_sampled = hard_bg[torch.randint(0,hard_bg.numel(),size=(hard_num,)).long()]
            bg_sampled = hard_bg_sampled
        else:
            raise NotImplementedError
        return bg_sampled




    @staticmethod
    def get_iou_with_same_class(rois,roi_labels,gt_boxes):
        """
        gt_boxes:(num of box per image,8)
        rois:(512,7)
        roi—labels
        """
        gt_labels = gt_boxes[:,7]
        gt_boxes = gt_boxes[:,:7]
        max_overlap = torch.zeros(rois.shape[0],dtype=torch.float32,device=gt_boxes.device)
        gt_assignment = torch.zeros(rois.shape[0],dtype=torch.long,device=gt_boxes.device)
        for idx in np.arange(gt_labels.min().item(),gt_labels.max().item()+1,1):
             rois_mask_class = roi_labels==idx
             # rois_mask_class_for_see = rois_mask_class.nonzero().view(-1)
             gt_mask_class = gt_labels == idx
             gt_origin_assign = torch.nonzero(gt_mask_class).view(-1)


             if gt_mask_class.sum()>0 and rois_mask_class.sum()>0:
                cur_gt = gt_boxes[gt_mask_class]
                cur_rois = rois[rois_mask_class]
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_rois,cur_gt)
                cur_max_overlap,cur_gt_assign = torch.max(iou3d,dim=1)
                # cur_max_overlaps_see = cur_max_overlap.cpu().numpy()
                max_overlap[rois_mask_class] =cur_max_overlap
                gt_assignment[rois_mask_class] = gt_origin_assign[cur_gt_assign]
        return max_overlap,gt_assignment



if __name__ == '__main__':
    a = np.random.rand(128)
    b = np.floor(a*512)
    print(a)
    #2.174,5.359