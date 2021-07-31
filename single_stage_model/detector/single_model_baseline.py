
import torch
import torch.nn as nn
import numpy as np
import os


from single_stage_model.backbone_3d.sparse_conv import Backbone3d
from single_stage_model.backbone_2d.backbone2d_module import Backbone2d

from single_stage_model.detect_head.head_2d import DetectHead
from single_stage_model.iou_head.iou_head_utils import IouHead

# from pvdet.ops.iou3d_nms import iou3d_nms_utils
from single_stage_model.iou3d_nms import iou3d_nms_utils
# from single_stage_model.configs.single_stage_config import cfg




class VEF(nn.Module):
    def __init__(self,num_channel=4):
        super().__init__()
        self.used_feature = num_channel
    def forward(self,voxels,num_per_voxel):
        mean_voxel = voxels.sum(dim=1)/ num_per_voxel.type_as(voxels).view(-1,1)
        return mean_voxel.contiguous()




class SSDbase(nn.Module):
    def __init__(self,logger,config,cfg):
        super().__init__()
        self.cfg = cfg
        self.config = config
        self.logger = logger
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.point_cloud_range = cfg.data_config.point_cloud_range

        self.feature_map_size = np.array((np.array(self.point_cloud_range [3:],dtype=np.float32)-np.array(self.point_cloud_range[:3],dtype=np.float32))\
                         / np.array(cfg.data_config.VoxelGenerator.voxel_size, dtype=np.float32),dtype=np.int)
        self.grid_size = cfg.data_config.VoxelGenerator.voxel_size
        self.spatial_shapes = np.array((self.feature_map_size[::-1] + [1,0,0]),dtype=np.int64)
        self.conv_3d = self.conv_2d= None
        self.using_iou_head = config["IouHead"]["enable"]
        self.class_names = cfg.CLASS_NAMES
        self.build_net(self.class_names,cfg)


    def build_net(self,class_names,cfg):
        self.vfe = VEF(cfg.data_config.num_used_features)
        self.conv_3d = Backbone3d(cfg.data_config.num_used_features,self.config["Conv3d"])
        self.conv_2d = Backbone2d()
        self.detect_head = DetectHead(grid_size=self.feature_map_size,config=cfg.model.detection_head,class_names = class_names)
        if self.using_iou_head:
            self.iou_head = IouHead(config=cfg.model.IouHead,in_channel=len(self.class_names)+7)

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            pass

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        if logger is not None:
            logger.info("****Load paramters from checkpoint %s to %s" % (filename, "CPU" if to_cpu else "GPU"))
        loc_type = torch.device("cpu") if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint["model_state"]

        if "version" in checkpoint:
            logger.info("===>checkpoint trained from version:%s" % checkpoint["version"])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info("Not update weight %s: %s" % (key, str(state_dict[key].shape)))
        logger.info("==>Done (load %d/%d)" % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info("==>Loading paramters from checkpoint %s to %s" % (filename, "CPU" if to_cpu else "GPU"))
        loc_type = torch.device("cpu") if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get("epoch", -1)
        it = checkpoint.get("it", 0.0)

        self.load_state_dict(checkpoint["model_state"])

        if optimizer is not None:
            if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
                logger.info("==>Loading optimizer parameters from checkpoint %s to %s"
                            % (filename, "CPU" if to_cpu else "GPU"))
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            else:
                assert filename[-4] == ".", filename
                src_file, ext = filename[:-4], filename[:-3]
                optimizer_filename = "%s_optim.%s" % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch





    def test_all_box_based_on_iou(self,batch_dict):
        """
        test all predicted box to find if there are valid boxes, but not selected based cls score
        """
        post_process_config = self.cfg.model["post_processing"]
        box_preds = batch_dict["rpn_box_preds"]
        cls_preds = batch_dict["rpn_cls_preds"]
        batch_size = batch_dict["batch_size"]
        gt_boxes = batch_dict["gt_boxes"]
        pred_dict = []
        recall_dict = {}
        for bs_id in range(batch_size):

            cur_box_preds = box_preds[bs_id]
            cur_cls_preds = cls_preds[bs_id]
            cur_gt = gt_boxes[bs_id]
            cnt = len(cur_gt) - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            if len(cur_gt)>0:
                # using anchor mask to filter
                # cur_box_preds = cur_box_preds[cur_anchor_mask]
                # cur_cls_preds = cur_cls_preds[cur_anchor_mask]

                # using iou value between gt and pred to filter
                cur_box_score, cur_cls_label = torch.max(cur_cls_preds, dim=1)
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_box_preds,cur_gt[:,:7])
                best_pred_box_id_for_per_gt = torch.argmax(iou3d,dim=0)
                iou3d_for_pred_and_gt = iou3d[best_pred_box_id_for_per_gt,torch.arange(len(cur_gt))]
                valid_mask = iou3d_for_pred_and_gt > 0.3
                best_pred_box_id_for_per_gt = best_pred_box_id_for_per_gt[valid_mask]
                cur_box_preds = cur_box_preds[best_pred_box_id_for_per_gt]
                cur_cls_label = cur_cls_label[best_pred_box_id_for_per_gt]
                cur_box_score = cur_box_score[best_pred_box_id_for_per_gt]

                selected_idx, selected_score = self.nms_of_box_proposal(cur_box_score, cur_box_preds, post_process_config)

                cur_box_preds = cur_box_preds[selected_idx]
                cur_cls_label = cur_cls_label[selected_idx]
                recall_dict = self.generate_recall_record(bs_id, cur_box_preds, batch_dict, recall_dict,
                                                          post_process_config.recall_thresh_list)
                single_batch_dict = {
                    "pred_boxes": cur_box_preds,
                    "pred_labels": cur_cls_label + 1,
                    "pred_scores": selected_score
                }
                pred_dict.append(single_batch_dict)
        return pred_dict,recall_dict

    def nms_based_iou(self,iou_score,boxes,config):
        src_iou_score = iou_score
        selected = []
        iou_thresh = config["iou_thresh"]
        nms_thresh = config["nms_thresh"]
        if iou_thresh >= 0:
            score_mask = iou_score >= iou_thresh
            iou_score = iou_score[score_mask]
            boxes = boxes[score_mask.squeeze(dim=-1)]
        if boxes.shape[0] > 0:
            pre_selected_score, indice = torch.topk(iou_score, k=min(config.pre_selection_num, boxes.shape[0]))
            box_for_nms = boxes[indice]
            after_nms_indx, _ = iou3d_nms_utils.nms_gpu(box_for_nms, pre_selected_score, nms_thresh)
            selected = indice[after_nms_indx[:config["post_selected_num"]]]

        if iou_thresh > 0:
            original_idx = score_mask.nonzero().view(-1)
            selected = original_idx[selected]
        return selected, src_iou_score[selected]

    def nms_of_box_proposal(self,box_score,boxes,config):
        src_box_scores = box_score
        selected = []
        score_thresh = config["cls_threshold"]
        nms_thresh = config["nms_thresh"]
        if score_thresh>0:
            score_mask = box_score>=score_thresh
            box_score = box_score[score_mask]
            boxes = boxes[score_mask]
        if boxes.shape[0]>0:
            pre_selected_score,indice = torch.topk(box_score,k=min(config.pre_selection_num,boxes.shape[0]))
            box_for_nms = boxes[indice]
            after_nms_indx,_ = iou3d_nms_utils.nms_gpu(box_for_nms,pre_selected_score,nms_thresh)
            selected = indice[after_nms_indx[:config["post_selected_num"]]]

        if score_thresh>0:
            original_idx = score_mask.nonzero().view(-1)
            selected = original_idx[selected]
        return selected, src_box_scores[selected]

    def post_processing_for_single_stage_model(self,batch_dict,tr=None,save_proposals=False):
        tr = self.config["post_processing"]["cls_threshold"]
        stratgy_id = self.config["post_processing"]["stratgy_id"]
        post_processing_stratgy = self.config["post_processing"]["stratgy_name"][stratgy_id]
        if post_processing_stratgy == "using_gt":
            pred_dict,recall_dict = self.test_all_box_based_on_iou(batch_dict)
            return pred_dict,recall_dict

        elif post_processing_stratgy == "using_iou":
            post_process_config = self.config["post_processing"]
            box_preds_selected = batch_dict["box_preds_selected"]
            cls_preds_selected = batch_dict["cls_preds_selected"]
            iou_preds_bin = batch_dict["iou_preds_bin"]
            iou_preds_residual = batch_dict["iou_preds_residual"]
            iou_preds_bin_id = torch.argmax(iou_preds_bin, dim=2).float()
            iou_preds = (iou_preds_bin_id.unsqueeze(dim=-1) * 0.2 + iou_preds_residual).squeeze(dim=-1)
            batch_size = batch_dict["batch_size"]
            pred_dict = []
            recall_dict = {}
            for bs_id in range(batch_size):
                cur_box_preds = box_preds_selected[bs_id]
                cur_cls_preds = cls_preds_selected[bs_id]
                cur_iou_preds = iou_preds[bs_id]
                cur_box_score, cur_cls_label = torch.max(cur_cls_preds, dim=1)
                # cur_box_score = torch.sigmoid(cur_box_score)

                selected_idx, selected_score = self.nms_based_iou(cur_iou_preds, cur_box_preds,
                                                                  post_process_config)

                cur_box_preds = cur_box_preds[selected_idx]
                cur_cls_label = cur_cls_label[selected_idx]

                recall_dict = self.generate_recall_record(bs_id, cur_box_preds, batch_dict, recall_dict,
                                                          post_process_config.recall_thresh_list)
                single_batch_dict = {
                    "pred_boxes": cur_box_preds,
                    "pred_labels": cur_cls_label + 1,
                    "pred_scores": selected_score
                }
                pred_dict.append(single_batch_dict)

        elif post_processing_stratgy == "using_class_score":
            post_process_config = self.cfg.model["post_processing"]
            box_preds = batch_dict["rpn_box_preds"]
            cls_preds = batch_dict["rpn_cls_preds"]
            batch_size = batch_dict["batch_size"]
            if batch_dict.get("anchor_masks", None):
                anchor_masks = torch.from_numpy(batch_dict["anchor_masks"]).cuda().permute(0, 2, 1).contiguous()
                anchor_masks = anchor_masks.view(batch_size, -1)

            pred_dict = []
            recall_dict = {}

            for bs_id in range(batch_size):
                cur_box_preds = box_preds[bs_id]
                cur_cls_preds = cls_preds[bs_id]
                if batch_dict.get("anchor_masks", None):
                    cur_anchor_mask = anchor_masks[bs_id]
                    cur_box_preds = cur_box_preds[cur_anchor_mask]
                    cur_cls_preds = cur_cls_preds[cur_anchor_mask]
                # using anchor mask to filter

                # using class score to filter
                cur_box_score, cur_cls_label = torch.max(cur_cls_preds, dim=1)
                cur_box_score = torch.sigmoid(cur_box_score)

                # if tr is not None:
                #     selected = cur_box_score > tr
                #     cur_box_score = cur_box_score[selected]
                #     cur_cls_label = cur_cls_label[selected]
                #     cur_box_preds = cur_box_preds[selected]
                # final nms
                selected_idx, selected_score = self.nms_of_box_proposal(cur_box_score, cur_box_preds,
                                                                        post_process_config)

                cur_box_preds = cur_box_preds[selected_idx]
                cur_cls_label = cur_cls_label[selected_idx]


                recall_dict = self.generate_recall_record(bs_id, cur_box_preds, batch_dict, recall_dict,
                                                          post_process_config.recall_thresh_list)
                single_batch_dict = {
                    "pred_boxes": cur_box_preds,
                    "pred_labels": cur_cls_label + 1,
                    "pred_scores": selected_score,
                }
                pred_dict.append(single_batch_dict)
        elif post_processing_stratgy == "cls_iou_blend":
            post_process_config = self.config["post_processing"]
            nms_thresh = post_process_config["nms_thresh"] # 0.1
            box_preds_selected = batch_dict["box_preds_selected"]
            cls_preds_selected = batch_dict["cls_preds_selected"]
            iou_preds_bin = batch_dict["iou_preds_bin"]
            iou_preds_residual = batch_dict["iou_preds_residual"]
            iou_preds_bin_id = torch.argmax(iou_preds_bin, dim=2).float()
            iou_preds = (iou_preds_bin_id.unsqueeze(dim=-1) * 0.2 + iou_preds_residual).squeeze(dim=-1)
            batch_size = batch_dict["batch_size"]

            if batch_dict.get("anchor_masks", None):
                anchor_masks = torch.from_numpy(batch_dict["anchor_masks"]).cuda().permute(0, 2, 1).contiguous()
                anchor_masks = anchor_masks.view(batch_size, -1)

            pred_dict = []
            recall_dict = {}

            for bs_id in range(batch_size):
                cur_box_preds = box_preds_selected[bs_id]
                cur_cls_preds = cls_preds_selected[bs_id]
                cur_iou_preds = iou_preds[bs_id]


                cur_box_score, cur_cls_label = torch.max(cur_cls_preds, dim=1)
                cur_box_score = torch.sigmoid(cur_box_score)

                if save_proposals:
                    box_proposals = cur_box_preds
                    proposals_cls = cur_cls_label +1
                    proposals_score = cur_box_score

                #selecte top k based cls score
                selected_score, indice = torch.topk(cur_box_score,k=min(self.config["post_processing"]["pre_selection_num"],cur_box_score.shape[0]) )
                cur_box_preds = cur_box_preds[indice]
                cur_cls_label = cur_cls_label[indice]
                cur_iou_preds = cur_iou_preds[indice]


                #using cls thresh to filter
                cls_thresh = self.config["post_processing"]["cls_threshold"]
                mask = selected_score>cls_thresh
                cur_box_preds = cur_box_preds[mask]
                selected_score = selected_score[mask]
                cur_cls_label = cur_cls_label[mask]
                cur_iou_preds = cur_iou_preds[mask]

                #nms
                after_nms_indx, _ = iou3d_nms_utils.nms_gpu(cur_box_preds, selected_score, nms_thresh)
                cur_box_preds = cur_box_preds[after_nms_indx]
                selected_score = selected_score[after_nms_indx]
                cur_cls_label = cur_cls_label[after_nms_indx]
                cur_iou_preds = cur_iou_preds[after_nms_indx]
                cur_cls_label = cur_cls_label +1
                # if 4 in cur_cls_label:
                #     bus_mask = cur_cls_label==4
                #     bus_box = cur_box_preds[bus_mask]
                #     bus_score = selected_score[bus_mask]
                #     bus_box_ = bus_box[:, [1, 0, 2, 3, 4, 5, 6]]
                #     bus_box_[:, 1] = -bus_box_[:, 1]
                #     bus_nms_indx,_ = iou3d_nms_utils.nms_gpu(bus_box_, bus_score, 0.0001)
                #     bus_box_ = bus_box_[bus_nms_indx]
                #
                #     IOU_3d = iou3d_nms_utils.boxes_iou3d_gpu(bus_box_,bus_box_)

                k_num = np.round(len(cur_iou_preds)*self.config["post_processing"]["topk_iou_ratio"]).astype(np.int)
                if k_num>=1:
                    cur_iou_preds,idx_by_iou = torch.topk(cur_iou_preds,k=int(k_num))
                    cur_box_preds = cur_box_preds[idx_by_iou]
                    selected_score = selected_score[idx_by_iou]
                    cur_cls_label = cur_cls_label[idx_by_iou]

                recall_dict = self.generate_recall_record(bs_id, cur_box_preds, batch_dict, recall_dict,
                                                          post_process_config.recall_thresh_list)

                single_batch_dict = {
                    "pred_boxes": cur_box_preds,
                    "pred_labels": cur_cls_label ,
                    "pred_scores": selected_score,
                    "box_proposals":box_proposals,
                    "proposals_cls":proposals_cls,
                    "proposals_score":proposals_score
                }
                pred_dict.append(single_batch_dict)

                # print("done")



        else:
            raise NotImplementedError

        return pred_dict,recall_dict






    def post_processing(self, batch_dict):
        post_process_config = self.cfg.model["post_processing"]
        box_preds = batch_dict["rpn_box_preds"]
        cls_preds = batch_dict["rpn_cls_preds"]
        batch_size = batch_dict["batch_size"]
        batch_dict= self.proposal_layer(batch_dict,post_process_config)

        pred_dicts = []
        recall_dict = {}
        for index in range(batch_size):
            cur_box_preds = box_preds[index]
            cur_cls_preds = cls_preds[index]
            cur_normal_cls_preds = torch.sigmoid(cur_cls_preds)
            label_preds = batch_dict["roi_labels"][index]
            selected, selected_score = self.class_final_nms(
                cur_normal_cls_preds,
                cur_box_preds,
                post_process_config.NMS_CONFIG,
                post_process_config.SCORE_THRESH)
            final_box = cur_box_preds[selected]
            final_score = selected_score
            final_cls = label_preds[selected]

            recall_dict = self.generate_recall_record(index, cur_box_preds, batch_dict, recall_dict,
                                                      post_process_config.RECALL_THRESH_LIST)
            single_batch_dict = {
                "pred_boxes": final_box,
                "pred_labels": final_cls,
                "pred_scores": final_score
            }

            pred_dicts.append(single_batch_dict)
        return pred_dicts, recall_dict

    def class_final_nms(self, box_scores, box_preds, nms_config, score_thresh=None):
        """
        box_scores:(N,)
        box_preds:(N,7)
        score thresh:(1,)
        """
        box_scores_src = box_scores
        score_mask = (box_scores > score_thresh).squeeze(dim=-1)
        box_scores = box_scores[score_mask]
        box_preds = box_preds[score_mask]
        selected = []
        if box_scores.shape[0] > 0:
            score_for_nms, indices = torch.topk(
                box_scores.squeeze(dim=-1), k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            box_for_nms = box_preds[indices]
            keep_id, _ = iou3d_nms_utils.nms_gpu(box_for_nms, score_for_nms, nms_config.NMS_THRESH)
            selected = indices[keep_id[:nms_config.NMS_POST_MAXSIZE]]

            origin_idx = torch.nonzero(score_mask).view(-1)
            selected = origin_idx[selected].view(-1)
        return selected, box_scores_src[selected].view(-1)

    def generate_recall_record(self, index, box_preds, batch_dict, recall_dict, thresh_list=(0.5, 0.7)):

        if batch_dict.get("gt_boxes",None) is not None:
            try:
                gt_boxes = batch_dict["gt_boxes"][index]
            except:
                print(batch_dict["sample_idx"][index])
                return recall_dict
            cur_gt = gt_boxes
            k = gt_boxes.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]

            if cur_gt.shape[0] > 0:
                if recall_dict.__len__() == 0:
                    recall_dict["gt"] = 0
                    for thresh in thresh_list:
                        recall_dict["roi_%s" % str(thresh)] = 0
                recall_dict["gt"] += cur_gt.shape[0]
                if len(box_preds)>0:
                    iou3d_rois = iou3d_nms_utils.boxes_iou3d_gpu(box_preds, cur_gt[:, :7])

                    for thresh in thresh_list:
                        rois_iou_max = iou3d_rois.max(dim=-1)[0]
                        recall_dict["roi_%s" % str(thresh)] += (rois_iou_max > thresh).sum().item()

        else:
            return recall_dict

        return recall_dict


if __name__ == '__main__':
    a = torch.tensor([

    ])
    f1 = a.view(-1)
    b = a.permute(1,0).contiguous()
    f2 = b.view(-1)
    assert f1 == f2
    print(f1)
    print(f2)