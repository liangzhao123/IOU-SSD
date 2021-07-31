import numpy as np
import numba
import io as sysio
from single_stage_model.dataset.leishen_dataset.tools.rotation_iou import rotate_iou_gpu_eval
# from single_stage_model.leishen_iou3d_nms import iou3d_nms_utils
from single_stage_model.iou3d_nms import iou3d_nms_utils
import torch
import time

# @numba.jit
def get_thresholds(scores:np.ndarray,num_gt,num_sample_pts =41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i,score in enumerate(scores):
        l_recall = (i+1)/num_gt
        if i <(len(scores)-1):
            r_recall = (i+2)/num_gt
        else:
            r_recall = l_recall
        if (r_recall - current_recall)<(current_recall - l_recall) and (i<len(scores) -1):
            continue
        thresholds.append(score)
        current_recall += 1/(num_sample_pts -1.0)
    return thresholds


def get_offical_eval_result(gt_annos,det_annos,current_classes,PR_detail_dict=None):
    overlap_0_7 = np.array([[0.7,0.5,0.5,0.7,0.5,0.7],
                            [0.7,0.5,0.5,0.7,0.5,0.7],
                            [0.7,0.5,0.5,0.7,0.5,0.7]])
    overlap_0_5 = np.array([[0.7,0.5,0.5,0.7,0.5,0.5],
                            [0.5,0.25,0.25,0.5,0.25,0.5],
                            [0.5,0.25,0.25,0.5,0.25,0.5]])
    min_overlaps = np.stack([overlap_0_7,overlap_0_5],axis=0) #（2,3,6）
    class_to_name = {
        0:"Car",
        1:"Pedestrian",
        2:"Rider",
        3:"Bus",
    }
    name_to_class = {v: n for n,v in class_to_name.items()}
    if not isinstance(current_classes,(list,tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls,str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int #[0,1,2]
    min_overlaps = min_overlaps[:,:,current_classes] #(2,3,3)
    result = ""
    #check whether aplpha is valid
    compute_aos = False
    # for anno in det_annos:
    #     if anno["alpha"].shape[0] !=0:
    #         if anno["alpha"][0] != -10:
    #             compute_aos =True
    #         break
    mAPbbox,mAPbev,mAP3d,mAPaos, mAPbbox_R40,mAPbev_R40,mAP3d_R40,mAPaos_R40 = do_eval(
        gt_annos,det_annos,current_classes,min_overlaps,compute_aos,PR_detail_dict
    )
    ret_dict = {}

    for j,curcls in enumerate(current_classes):
        for i in range(min_overlaps.shape[0]):
            result += print_str(f"{class_to_name[curcls]} " 
                                "AP@{:.2f}, {:.2f},{:.2f}:".format(*min_overlaps[i,:,j]))
            result += print_str((f"bbox AP:{mAPbbox[j,0,i]:.4f},"
                                 f"{mAPbbox[j,1,i]:.4f},"
                                 f"{mAPbbox[j,2,i]:.4f}"))
            result += print_str((f"bev  AP:{mAPbev[j,0,i]:.4f},"
                                 f"{mAPbev[j,1,i]:.4f},"
                                 f"{mAPbev[j,2,i]:.4f}"))
            result += print_str((f"3d   AP:{mAP3d[j,0,i]:.4f},"
                                 f"{mAP3d[j,1,i]:.4f},"
                                 f"{mAP3d[j,2,i]:.4f},"))
            if compute_aos:
                result += print_str((f"aos  AP:{mAPaos[j,0,i]:.4f}, "
                                     f"{mAPaos[j,1,i]:.4f}, "
                                     f"{mAPaos[j,2,i]:.4f}"))
                if i ==0 :
                    ret_dict["%s_aos_easy" % class_to_name[curcls]] = mAPaos[j,0,0]
                    ret_dict["%s_aos_moderate" % class_to_name[curcls]] = mAPaos[j,1,0]
                    ret_dict["%s_aos_hard" % class_to_name[curcls]] = mAPaos[j,2,0]
            result += print_str((f"{class_to_name[curcls]} "
                                 "AP_R40@{:.2f}, {:.2f}, {:.2f}: ".format(*min_overlaps[i,:,j])))
            result += print_str(f"bbox AP:{mAPbbox_R40[j,0,i]:.4f},"
                                f"{mAPbbox_R40[j,1,i]:.4f},"
                                f"{mAPbbox_R40[j,2,i]:.4f}")
            result += print_str(f"bev AP:{mAPbev_R40[j, 0, i]:.4f},"
                                f"{mAPbev_R40[j, 1, i]:.4f},"
                                f"{mAPbev_R40[j, 2, i]:.4f}")
            result += print_str(f"3d AP:{mAP3d_R40[j, 0, i]:.4f},"
                                f"{mAP3d_R40[j, 1, i]:.4f},"
                                f"{mAP3d_R40[j, 2, i]:.4f}")

            if compute_aos:
                result += print_str((f"aos R40  AP:{mAPaos_R40[j,0,i]:.4f}, "
                                     f"{mAPaos_R40[j,1,i]:.4f}, "
                                     f"{mAPaos_R40[j,2,i]:.4f}"))
                if i ==0 :
                    ret_dict["%s_aos_easy_R40" % class_to_name[curcls]] = mAPaos_R40[j,0,0]
                    ret_dict["%s_aos_moderate_R40" % class_to_name[curcls]] = mAPaos_R40[j,1,0]
                    ret_dict["%s_aos_hard_R40" % class_to_name[curcls]] = mAPaos_R40[j,2,0]


            if i == 0:
                ret_dict["%s_3d_easy" % class_to_name[curcls]] = mAP3d[j,0,0]
                ret_dict["%s_3d_moderate" % class_to_name[curcls]] = mAP3d[j,1,0]
                ret_dict["%s_3d_hard" % class_to_name[curcls]] = mAP3d[j,2,0]
                ret_dict['%s_bev_easy' % class_to_name[curcls]] = mAPbev[j, 0, 0]
                ret_dict['%s_bev_moderate' % class_to_name[curcls]] = mAPbev[j, 1, 0]
                ret_dict['%s_bev_hard' % class_to_name[curcls]] = mAPbev[j, 2, 0]
                ret_dict['%s_image_easy' % class_to_name[curcls]] = mAPbbox[j, 0, 0]
                ret_dict['%s_image_moderate' % class_to_name[curcls]] = mAPbbox[j, 1, 0]
                ret_dict['%s_image_hard' % class_to_name[curcls]] = mAPbbox[j, 2, 0]

                ret_dict['%s_3d_easy_R40' % class_to_name[curcls]] = mAP3d_R40[j, 0, 0]
                ret_dict['%s_3d_moderate_R40' % class_to_name[curcls]] = mAP3d_R40[j, 1, 0]
                ret_dict['%s_3d_hard_R40' % class_to_name[curcls]] = mAP3d_R40[j, 2, 0]
                ret_dict['%s_bev_easy_R40' % class_to_name[curcls]] = mAPbev_R40[j, 0, 0]
                ret_dict['%s_bev_moderate_R40' % class_to_name[curcls]] = mAPbev_R40[j, 1, 0]
                ret_dict['%s_bev_hard_R40' % class_to_name[curcls]] = mAPbev_R40[j, 2, 0]
                ret_dict['%s_image_easy_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 0, 0]
                ret_dict['%s_image_moderate_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 1, 0]
                ret_dict['%s_image_hard_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 2, 0]
    return result,ret_dict


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()




def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            compute_aos =False,
            PR_detail_dict = None):
    difficultys =[0,1,2]
    ret = eval_class(gt_annos,dt_annos,current_classes,
                     difficultys,2,min_overlaps,compute_aos,)
    mAP_bbox = get_mAP(ret["precision"])
    mAP_bbox_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict["bbox"] = ret["precision"]

    mAP_aos = mAP_aos_R40 = None
    if compute_aos:
        mAP_aos = get_mAP(ret["orientation"])
        mAP_aos_R40 = get_mAP_R40(ret["orientation"])
        if PR_detail_dict is not None:
            PR_detail_dict["aos"] = ret["orientation"]
    ret = eval_class(gt_annos,dt_annos,current_classes,difficultys,1,min_overlaps)

    mAP_bev = get_mAP(ret["precision"])
    mAP_bev_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict["bev"] = ret["precision"]
    ret = eval_class(gt_annos,dt_annos,current_classes,difficultys,2,min_overlaps)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict["3d"] = ret["precision"]
    return mAP_bbox,mAP_bev,mAP_3d,mAP_aos, mAP_bbox_R40,mAP_bev_R40,mAP_3d_R40,mAP_aos_R40


def get_mAP_R40(prec):
    sums =0
    for i in range(1,prec.shape[-1]):
        sums = sums+ prec[...,i]
    return sums / 40*100



def get_mAP(prec):
    sums =0
    for i in range(0,prec.shape[-1],4):
        sums = sums + prec[...,i]
    return sums /11*100


# @numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[gt_num:gt_num+gt_nums[i],dt_num:dt_num+dt_nums[i]]
            gt_data = gt_datas[gt_num:gt_num+gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num+dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp,fp,fn,simliarity,_ = compute_statistic_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos
            )
            pr[t,0] += tp
            pr[t,1] += fp
            pr[t,2] += fn
            if simliarity != -1:
                pr[t,3] +=simliarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]






def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos =False,
               num_parts = 50):
    """
    :param gt_annos:真实值
    :param dt_annos:预测值
    :param current_classes: 0:car,1:pedestrian,2:cyclist
    :param difficultys: 0:easy,1:normal,2:cyclist
    :param metric: 0:bbox,1:bev,2:3d
    :param min_overlaps: [min_overlap,metric,class]
    :param compute_aos:
    :param num_parts: 快速计算
    :return:
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples,num_parts)

    rets = calculate_iou_partly(gt_annos,dt_annos,metric,num_parts)
    overlaps, parted_overlaps, total_gt_num, total_dt_num = rets



    N_SAMPLED_POINTS = 41 #这个值的选取为什么是选死的
    num_min_overlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros([num_class,num_difficulty,num_min_overlap,N_SAMPLED_POINTS])
    recall = np.zeros([num_class,num_difficulty,num_min_overlap,N_SAMPLED_POINTS])
    aos = np.zeros([num_class,num_difficulty,num_min_overlap,N_SAMPLED_POINTS])
    print("Metric:", metric)
    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            rets = _prepar_data(gt_annos,dt_annos,current_class,difficulty)
            (gt_datas_list,dt_datas_list,ignored_gts,ignored_dets,
            dontcares,total_dc_num,total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:,metric,m]):
                print("min_overlap: ",min_overlap)
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistic_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap = min_overlap,
                        thresh = 0.0,
                        compute_fp = False,
                        compute_aos = False
                    )

                    tp,fp,fn,similarity,thresholds = rets#这里的thresholds（是每一帧下的gt的个数，）对应的值是dt score
                    thresholdss += thresholds.tolist()

                thresholdss = np.array(thresholdss)

                thresholds = get_thresholds(thresholdss,total_num_valid_gt)
                thresholds = np.array(thresholds)

                pr = np.zeros([len(thresholds),4])

                idx = 0
                for j,num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx+num_part],0
                    )
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx+num_part],0
                    )
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx+num_part],0
                    )
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx+num_part],0
                    )
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0
                    )

                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx+ num_part],
                        total_dt_num[idx:idx+ num_part],
                        total_dc_num[idx:idx+ num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap = min_overlap,
                        thresholds = thresholds,
                        compute_aos = compute_aos
                    )
                    idx += num_part
                for i in range(len(thresholds)):
                    recall[m,l,k,i]= pr[i,0]/(pr[i,0]+pr[i,2])#tp/(tp+fn)
                    precision[m,l,k,i] = pr[i,0]/(pr[i,0]+pr[i,1])#tp/(tp+fp)
                    if compute_aos:
                        aos[m,l,k,i] = pr[i,3]/(pr[i,0]+pr[i,1])
                for i in range(len(thresholds)):
                    precision[m,l,k,i] = np.max(
                        precision[m,l,k,i:],axis=-1
                    )
                    recall[m,l,k,i] = np.max(
                        recall[m,l,k,i:],axis=-1
                    )
                    if compute_aos:
                        aos[m,l,k,i] = np.max(
                            aos[m,l,k,i:],axis=-1
                        )
    ret_dict = {
        "recall":recall,
        "precision":precision,
        "orientation":aos
    }
    return ret_dict







# @numba.jit(nopython=True)
def compute_statistic_jit(overlaps,
                          gt_datas,
                          dt_datas,
                          ignored_gt,
                          ignored_det,
                          dc_bboxes,
                          metric,
                          min_overlap,
                          thresh=0,
                          compute_fp = False,
                          compute_aos = False):
    """
    这个函数是计算recall和precision的核心函数
    :param overlaps: 这个是一帧下的gt和dt计算的overlaps
    :param gt_datas: 这个是一帧的gt里面是{bbox，alpha，score}
    :param dt_datas: 这个是一帧的dt，里面是{bbox，alpha，score}
    :param ignored_gt:忽略是1,其他为0或者-1，这个也是对应的一个frame下的gt
    :param ignored_det:忽略为1,其他为0，，这个也是对应的一个frame下的dt
    :param dc_bboxes:这个name为dontcare的box{}
    :param metric:0：bbox,1:bev,2:3d
    :param min_overlap:是一个arrray，shape为（2,1,1）
    :param thresh:
    :param compute_fp:
    :param compute_aos:
    :return:thresholds:这个是每一帧下的每个gt对应的最好的dt 的dt score也就是rcnn net里的preds cls这个值,
                        训练的target是（0-1） roi_iou大于0.75为1,在0.75到0.25之间为0-1之间的数，roi_iou小于0.25则target为0，
                        所以可以看出这个值iou值
    """

    det_size = dt_datas.shape[0]#每一帧下det检测的目标的个数
    gt_size = gt_datas.shape[0]#每一帧下gt的目标的个数
    dt_scores = dt_datas[:,-1]
    dt_alphas = np.zeros(dt_scores.shape)
    gt_alphas = np.zeros(dt_scores.shape)
    # dt_alphas = dt_datas[:,4]
    # gt_alphas = gt_datas[:,4]

    # dt_bboxes = dt_datas[:,:4]#每一帧下检测目标的bbox(x min,ymin.xmax,ymax)
    # gt_bboxes = gt_datas[:,:4]#每一帧下gt的目标的bbox(x min,ymin.xmax,ymax)
    dt_bboxes = dt_datas.copy()

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size

    #过滤掉dt score中小于0的dt box
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i]<thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size,)) #每个gt都有应该是匹配的dt里的最好的dt score，但这个dt score基本上是0附近



    thresh_idx =0
    delta = np.zeros((gt_size,))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if ignored_det[j]==-1:
                continue
            if assigned_detection[j]:
                continue
            if ignored_threshold[j]:
                continue
            overlap = overlaps[i,j] #

            dt_score = dt_scores[j]
            if (not compute_fp and (overlap> min_overlap)#后边的elif没用因为这里直接compute fp为False
                 and (dt_score > valid_detection)):       #所以这里就是找到gt对应的dt里面的最大的dt score和对应的id
                det_idx = j                               #其实这里可以使用numpy的函数代替
                valid_detection = dt_score
            elif (compute_fp and (overlap>min_overlap) and (overlap>max_overlap or assigned_ignored_det)):
                max_overlap = overlap
                det_idx = j
                valid_detection =1
                assigned_ignored_det = False
            elif (compute_fp and (overlap>min_overlap)
                  and (valid_detection==NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection =1
                assigned_ignored_det =True

        if (valid_detection == NO_DETECTION and ignored_gt[i]==0):
            fn += 1
        elif ((valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx]==1)):
            assigned_detection[det_idx] = True

        elif valid_detection != NO_DETECTION:
            tp +=1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx +=1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        elif metric==2:
            if dc_bboxes.shape[0]>0:
                overlaps_dt_dc = iou3d_nms_utils.boxes_iou3d_gpu(torch.from_numpy(dt_bboxes.astype(np.float32)).cuda(),
                                                           torch.from_numpy(dc_bboxes.astype(np.float32)).
                                                           cuda()).cpu().numpy().astype(np.float64)
                for i in range(dc_bboxes.shape[0]):
                    for j in range(det_size):
                        if (assigned_detection[j]):
                            continue
                        if (ignored_det[j] == -1 or ignored_det[j] == 1):
                            continue
                        if (ignored_threshold[j]):
                            continue
                        if overlaps_dt_dc[j, i] > min_overlap:
                            assigned_detection[j] = True
                            nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1


    return tp,fp,fn,similarity,thresholds[:thresh_idx]




















def _prepar_data(gt_annos,dt_annos,current_class,difficulty):
    dt_data_list = []
    gt_data_list = []
    total_dc_num = []
    ignored_gts, ignored_dets,dontcares = [], [],[]
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i],dt_annos[i],current_class,difficulty)
        num_valid_gt,ignored_gt,ignored_det,dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt,dtype=np.int64))
        ignored_dets.append(np.array(ignored_det,dtype = np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0,4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes,0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        # gt_datas = np.concatenate(
        #     [gt_annos[i]["bbox"],gt_annos[i]["alpha"][...,np.newaxis],
        #      gt_annos[i]["score"][...,np.newaxis]],1)
        # dt_datas = np.concatenate(
        #     [dt_annos[i]["bbox"],dt_annos[i]["alpha"][...,np.newaxis],
        #      dt_annos[i]["score"][...,np.newaxis]],1)
        # gt_data_list.append(gt_datas)
        # dt_data_list.append(dt_datas)

        dt_datas = np.concatenate([dt_annos[i]['boxes_lidar'], dt_annos[i]["score"][..., np.newaxis]], 1)
        gt_datas = np.array(gt_annos[i]['gt_boxes_lidar'])


        dt_data_list.append(dt_datas)
        gt_data_list.append(gt_datas)

    total_dc_num = np.stack(total_dc_num,axis=0)
    return (gt_data_list,dt_data_list,ignored_gts,ignored_dets,dontcares,
            total_dc_num,total_num_valid_gt)




def clean_data(gt_anno,dt_anno,current_class,difficulty):
    CLASS_NAMES = ["car","pedestrian","rider","bus"]
    # MIN_HEIGHT = [40,25,25]
    MIN_RANGE = [10, 20, 35]
    MAX_OCCLUSION = [0,1,2]
    MAX_TRUNCATION = [0.15,0.3,0.5]
    dc_bboxes, ignored_gt, ignored_dt = [],[],[]
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0

    for i in range(num_gt):
        # bbox = gt_anno["bbox"][i]
        gt_box3d = gt_anno['gt_boxes_lidar'][i]
        gt_name = gt_anno["name"][i].lower()
        # dis = np.abs(gt_box3d[1])
        dis = np.sqrt(gt_box3d[0]**2+gt_box3d[1]**2+gt_box3d[2]**2)# y is the forward direction in leishen
        valid_class = -1
        if gt_name == current_cls_name:
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower()==gt_name):
            valid_class = 0
        else:
            valid_class =-1
        ignore = False
        # if ((gt_anno["occluded"][i]>MAX_OCCLUSION[difficulty])
        # or (gt_anno["truncated"][i]>MAX_TRUNCATION[difficulty])
        # or (height <= MIN_HEIGHT[difficulty])):
        #     ignore=True
        if difficulty == 0:
            if dis > MIN_RANGE[0]:
                ignore = True
        elif difficulty == 1:
            lower,up = MIN_RANGE[0],MIN_RANGE[1]
            # if dis <= lower or dis >= up:
            if dis > MIN_RANGE[1]:
                ignore = True
        elif difficulty == 2:
            if dis > MIN_RANGE[2]:
                ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt +=1
        elif valid_class ==0 or (ignore and valid_class == 1):
            ignored_gt.append(1)
        else :
            ignored_gt.append(-1)
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        # height = abs(dt_anno["bbox"][i,3]-dt_anno["bbox"][i,1])
        dt_box3d = dt_anno["boxes_lidar"][i]
        dis = np.sqrt(dt_box3d[0]**2+dt_box3d[1]**2+dt_box3d[2]**2)

        # if height < MIN_HEIGHT[difficulty]:
        #     ignored_dt.append(1)
        # elif valid_class ==1:
        #     ignored_dt.append(0)
        # else:
        #     ignored_dt.append(-1)

        if difficulty ==0:
            if dis > MIN_RANGE[0]:
                ignored_dt.append(1)
            elif valid_class ==1:
                ignored_dt.append(0)
            else:
                ignored_dt.append(-1)
        elif difficulty ==1:
            lower, up = MIN_RANGE[0], MIN_RANGE[1]
            # if dis <= lower or dis >= up:
            if dis >= up:
                ignored_dt.append(1)
            elif valid_class ==1:
                ignored_dt.append(0)
            else:
                ignored_dt.append(-1)
        elif difficulty ==2:
            if dis > MIN_RANGE[2]:
                ignored_dt.append(1)
            elif valid_class ==1:
                ignored_dt.append(0)
            else:
                ignored_dt.append(-1)
    return num_valid_gt,ignored_gt,ignored_dt,dc_bboxes












def calculate_iou_partly(gt_annos,dt_annos,metric,num_parts=50):
    """
    这个函数是快速的 计算 iou的函数，但是是相机坐标系下的
    :param gt_annos:
    :param dt_annos:
    :param metric: 0:bbox,1:bev,2:3d
    :param num_parts: 快速计算的一个参数
    :return:
    """
    assert len(gt_annos) ==len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos],0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos],0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples,num_parts)
    parted_overlaps = []
    example_idx = 0



    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx+num_part]
        dt_annos_part = dt_annos[example_idx:example_idx+num_part]
        if metric ==0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part],0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part],0)
            overlap_part = image_box_overlap(gt_boxes,dt_boxes)
        elif metric ==1:
            # loc = np.concatenate(
            #     [a["location"][:,[0,2]] for a in gt_annos_part],0)
            # dims = np.concatenate(
            #     [a["dimensions"][:,[0,2]] for a in gt_annos_part ],0)
            # rots = np.concatenate([a["rotation_y"] for a in gt_annos_part],0)
            # gt_boxes = np.concatenate([loc,dims,rots[...,np.newaxis]],axis=1)
            gt_boxes3d_lidar = np.concatenate([a['gt_boxes_lidar'] for a in gt_annos_part], 0)
            gt_bev_boxes = gt_boxes3d_lidar[:,[0,1,3,4,6]]

            # loc = np.concatenate(
            #     [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            # dims = np.concatenate(
            #     [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            # rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            # dt_boxes = np.concatenate(
            #     [loc,dims,rots[...,np.newaxis]],axis=1)
            dt_boxes_3d = np.concatenate([a["boxes_lidar"] for a in dt_annos_part], 0)
            dt_bev_boxes = dt_boxes_3d[:,[0,1,3,4,6]]
            # overlap_part = bev_box_overlap(gt_bev_boxes,dt_bev_boxes).astype(np.float64)
            overlap_part = iou3d_nms_utils.boxes_iou_bev(torch.from_numpy(gt_boxes3d_lidar.astype(np.float32)).cuda(),
                                                           torch.from_numpy(dt_boxes_3d.astype(np.float32)).
                                                           cuda()).cpu().numpy().astype(np.float64)
        elif metric == 2:
            # loc = np.concatenate([a["location"] for a in gt_annos_part],0)
            # dims = np.concatenate([a["dimensions"] for a in gt_annos_part],0)
            # rots = np.concatenate([a["rotation_y"] for a in gt_annos_part],0)
            # gt_boxes = np.concatenate(
            #     [loc,dims,rots[...,np.newaxis]],1
            # )
            gt_boxes = np.concatenate([a['gt_boxes_lidar'] for a in gt_annos_part], 0)

            try :
                loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
                dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
                dt_boxes = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], 1
                )
            except:
                dt_boxes = np.concatenate([a["boxes_lidar"] for a in dt_annos_part],0)
            # overlap_part = d3_box_overlap(gt_boxes,dt_boxes).astype(np.float64)
            overlap_part = iou3d_nms_utils.boxes_iou3d_gpu(torch.from_numpy(gt_boxes.astype(np.float32)).cuda(),torch.from_numpy(dt_boxes.astype(np.float32)).
                                                           cuda()).cpu().numpy().astype(np.float64)

        else:
            raise ValueError("unknown error")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
        #break#测试用
    overlaps = []
    example_idx = 0
    for j,num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(parted_overlaps[j][gt_num_idx:gt_num_idx+gt_box_num,
                            dt_num_idx:dt_num_idx+dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
            #break#测试
        example_idx += num_part
        #break  # 测试

    return overlaps,parted_overlaps,total_gt_num,total_dt_num


def bev_box_overlap(gt_boxes,dt_boxes,criterion=-1):
    riou = rotate_iou_gpu_eval(gt_boxes,dt_boxes,criterion)
    return riou


def d3_box_overlap(gt_boxes,dt_boxes,criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes=gt_boxes[:,[0,2,3,5,6]],
                               query_boxes=dt_boxes[:,[0,2,3,5,6]],criterion=2)
    d3_box_overlap_kernel(gt_boxes,dt_boxes,rinc,criterion)
    return rinc


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]#平面乘以高度信息就是真个3d的inter部分
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0

@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    """
    :param boxes:
    :param query_boxes:
    :param criterion:  0:bbox,1:bev,2:3d
    :return:
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps



def get_split_parts(num,num_part):
    same_part = num//num_part
    remain_num = num % num_part
    if same_part ==0 :
        return [num]
    if remain_num ==0 :
        return [same_part] * num_part
    else:
        return [same_part]*num_part +[remain_num]






