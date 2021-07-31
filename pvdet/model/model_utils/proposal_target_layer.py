import numpy as np
import torch
from pvdet.ops.iou3d_nms import iou3d_nms_utils
from pvdet.tools.config import cfg
import torch.nn as nn

def proposal_target_layer(input_dict, roi_sampler_cfg):
    rois = input_dict['rois'] #
    roi_raw_scores = input_dict['roi_raw_scores']#这个实际是rpn head里的cls pred进行view以后（B,num_anchor,3）然后又取里每一列的最大值
    roi_labels = input_dict['roi_labels']#这个实际是rpn head里的cls pred进行view以后（B,num_anchor,3）然后又取里每一列的最大值的编号+1
    gt_boxes = input_dict['gt_boxes']#这个是gt box（B,M,7+1）1是class类别

    batch_rois, batch_gt_of_rois, batch_roi_iou, batch_roi_raw_scores, batch_roi_labels = \
        sample_rois_for_rcnn(rois, gt_boxes, roi_raw_scores, roi_labels, roi_sampler_cfg)
    reg_valid_mask = (batch_roi_iou > roi_sampler_cfg.REG_FG_THRESH).long()#0.55

    if roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
        batch_cls_label = (batch_roi_iou > roi_sampler_cfg.CLS_FG_THRESH).long()
        invalid_mask = (batch_roi_iou > roi_sampler_cfg.CLS_BG_THRESH) & \
                       (batch_roi_iou < roi_sampler_cfg.CLS_FG_THRESH)#0.25<x<0.75
        batch_cls_label[invalid_mask>0] = -1
    elif roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou':
        fg_mask = batch_roi_iou > roi_sampler_cfg.CLS_FG_THRESH#0.75
        bg_mask = batch_roi_iou < roi_sampler_cfg.CLS_BG_THRESH#0.25
        interval_mask = (fg_mask == 0) & (bg_mask == 0)#(B,128)

        batch_cls_label = (fg_mask > 0).float()#(B,128)
        batch_cls_label[interval_mask] = batch_roi_iou[interval_mask] * 2 - 0.5 #缩放到0-1.0之间
    else:
        raise NotImplementedError
    output_dict = {'rcnn_cls_labels': batch_cls_label.view(-1),#(B,128)iou>0.25小于0.75
                   'reg_valid_mask': reg_valid_mask.view(-1),#(B,128)iou>0.55的
                   'gt_of_rois': batch_gt_of_rois,#(B,128,7+1)+1是class
                   'gt_iou': batch_roi_iou,#(B,128)是roi和gt的iou值
                   'rois': batch_rois,#(B,128,7)这里的7代表的真实的[x,y,z,w,l,h,ry]真实值不是相对值
                   'roi_raw_scores': batch_roi_raw_scores,
                   'roi_labels': batch_roi_labels}
    return output_dict


def sample_rois_for_rcnn(roi_boxes3d,
                         gt_boxes3d,
                         roi_raw_scores,
                         roi_labels, roi_sampler_cfg):

    """
    :param roi_boxes3d: (B,512,7)
    :param gt_boxes3d: (B,M,7+1)这个1是box的种类（“car”,1,"Pedestrian",2,"Cyclist",3）
    :param roi_raw_scores:(B,512)
    :param roi_labels:(B,512)
    :param roi_samlpler_cfg:
    :return:
    """
    batch_size = roi_boxes3d.size(0)

    fg_rois_per_image = int(np.round(roi_sampler_cfg.FG_RATIO * roi_sampler_cfg.ROI_PER_IMAGE))

    code_size = roi_boxes3d.shape[-1]
    batch_rois = gt_boxes3d.new(batch_size, roi_sampler_cfg.ROI_PER_IMAGE, code_size).zero_()
    batch_gt_of_rois = gt_boxes3d.new(batch_size, roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1).zero_()
    batch_roi_iou = gt_boxes3d.new(batch_size, roi_sampler_cfg.ROI_PER_IMAGE).zero_()
    batch_roi_raw_scores = gt_boxes3d.new(batch_size, roi_sampler_cfg.ROI_PER_IMAGE).zero_()
    batch_roi_labels = gt_boxes3d.new(batch_size, roi_sampler_cfg.ROI_PER_IMAGE).zero_().long()

    for idx in range(batch_size):
        cur_roi, cur_gt, cur_roi_raw_scores, cur_roi_labels = \
            roi_boxes3d[idx], gt_boxes3d[idx], roi_raw_scores[idx], roi_labels[idx]

        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if len(cfg.CLASS_NAMES) == 1:
            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
        else:
            cur_gt_labels = cur_gt[:, -1].long()
            max_overlaps, gt_assignment = get_maxiou3d_with_same_class(cur_roi, cur_roi_labels,
                                                                       cur_gt[:, 0:7], cur_gt_labels)

        fg_thresh = min(roi_sampler_cfg.REG_FG_THRESH, roi_sampler_cfg.CLS_FG_THRESH)#0.55
        fg_inds = torch.nonzero((max_overlaps >= fg_thresh)).view(-1)
        #torch.nonzero这个函数是返回非0的元素的编号，这里返回的是roi的编号
        easy_bg_inds = torch.nonzero((max_overlaps < roi_sampler_cfg.CLS_BG_THRESH_LO)).view(-1)#小于0.1的是easy bg
        hard_bg_inds = torch.nonzero((max_overlaps < roi_sampler_cfg.REG_FG_THRESH) &
                                     (max_overlaps >= roi_sampler_cfg.CLS_BG_THRESH_LO)).view(-1)#在0.1到0.55之间的是hard

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes3d).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, roi_sampler_cfg)

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes3d).long()
            fg_inds = fg_inds[rand_num]
            fg_rois_per_this_image = roi_sampler_cfg.ROI_PER_IMAGE
            bg_rois_per_this_image = 0
        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, roi_sampler_cfg)

            fg_rois_per_this_image = 0
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        roi_list, roi_iou_list, roi_gt_list, roi_score_list, roi_labels_list = [], [], [], [], []
        if fg_rois_per_this_image > 0:
            fg_rois = cur_roi[fg_inds]
            gt_of_fg_rois = cur_gt[gt_assignment[fg_inds]]
            fg_iou3d = max_overlaps[fg_inds]

            roi_list.append(fg_rois)
            roi_iou_list.append(fg_iou3d)
            roi_gt_list.append(gt_of_fg_rois)
            roi_score_list.append(cur_roi_raw_scores[fg_inds])
            roi_labels_list.append(cur_roi_labels[fg_inds])

        if bg_rois_per_this_image > 0:
            bg_rois = cur_roi[bg_inds]
            gt_of_bg_rois = cur_gt[gt_assignment[bg_inds]]
            bg_iou3d = max_overlaps[bg_inds]

            roi_list.append(bg_rois)
            roi_iou_list.append(bg_iou3d)
            roi_gt_list.append(gt_of_bg_rois)
            roi_score_list.append(cur_roi_raw_scores[bg_inds])
            roi_labels_list.append(cur_roi_labels[bg_inds])

        rois = torch.cat(roi_list, dim=0)
        iou_of_rois = torch.cat(roi_iou_list, dim=0)
        gt_of_rois = torch.cat(roi_gt_list, dim=0)
        cur_roi_raw_scores = torch.cat(roi_score_list, dim=0)
        cur_roi_labels = torch.cat(roi_labels_list, dim=0)

        batch_rois[idx] = rois
        batch_gt_of_rois[idx] = gt_of_rois
        batch_roi_iou[idx] = iou_of_rois
        batch_roi_raw_scores[idx] = cur_roi_raw_scores
        batch_roi_labels[idx] = cur_roi_labels

    return batch_rois, batch_gt_of_rois, batch_roi_iou, batch_roi_raw_scores, batch_roi_labels

def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, roi_sampler_cfg):
    if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
        hard_bg_rois_num = int(bg_rois_per_this_image * roi_sampler_cfg.HARD_BG_RATIO)
        easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

        # sampling hard bg
        rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
        hard_bg_inds = hard_bg_inds[rand_idx]

        # sampling easy bg
        rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
        easy_bg_inds = easy_bg_inds[rand_idx]

        bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
    elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
        hard_bg_rois_num = bg_rois_per_this_image
        # sampling hard bg
        rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
        bg_inds = hard_bg_inds[rand_idx]
    elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
        easy_bg_rois_num = bg_rois_per_this_image
        # sampling easy bg
        rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
        bg_inds = easy_bg_inds[rand_idx]
    else:
        raise NotImplementedError

    return bg_inds

def get_maxiou3d_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
    """
    三类物体的rois与真实的gt boxes进行匹配，得到每个roi最匹配的gt boxe的indx
    :param rois: (N, 7)
    :param roi_labels: (N)
    :param gt_boxes: (N, 8)
    :return:
    """
    max_overlaps = rois.new_zeros(rois.shape[0])
    gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

    for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
        roi_mask = (roi_labels == k)
        gt_mask = (gt_labels == k)
        if roi_mask.sum() > 0 and gt_mask.sum() > 0:
            cur_roi = rois[roi_mask]
            cur_gt = gt_boxes[gt_mask]
            original_gt_assignment = gt_mask.nonzero().view(-1)

            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
            cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
            max_overlaps[roi_mask] = cur_max_overlaps
            gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

    return max_overlaps, gt_assignment



class ProposalTargetLayer(nn.Module):
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """
        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(
            batch_dict=batch_dict
        )
        # regression valid mask
        reg_valid_mask = (batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()

        # classification label
        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
            batch_cls_labels = (batch_roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH).long()
            ignore_mask = (batch_roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & \
                          (batch_roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            batch_cls_labels[ignore_mask > 0] = -1
        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou':
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = batch_roi_ious > iou_fg_thresh
            bg_mask = batch_roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            batch_cls_labels = (fg_mask > 0).float()
            batch_cls_labels[interval_mask] = \
                (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        else:
            raise NotImplementedError

        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels}

        return targets_dict

    def sample_rois_for_rcnn(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_boxes']

        code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1)
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)

        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
                rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]
            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi, roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                )
            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)

            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_ious[index] = max_overlaps[sampled_inds]
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]

        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels

    def subsample_rois(self, max_overlaps):
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)

        fg_inds = torch.nonzero((max_overlaps >= fg_thresh)).view(-1)
        easy_bg_inds = torch.nonzero((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).view(-1)
        hard_bg_inds = torch.nonzero((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) &
                                     (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = []

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1)

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
            boxes_for_nms, box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]
