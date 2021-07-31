import tqdm
import time
import pickle
from new_train.config import cfg
from pvdet.tools.train_utils.train_utils import example_convert_to_torch
import os


def eval_one_epoch(model,
                   data_loader,
                   epoch_id,
                   logger,
                   save_to_file=False,
                   result_dir=None,
                   test_mode = False):
    os.makedirs(result_dir,exist_ok=True)

    if save_to_file:
        final_output_dir = os.path.join(result_dir,"final_result","data")
        os.makedirs(final_output_dir,exist_ok=True)
    else:
        final_output_dir = None
    metric = {
        "num_gt":0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric["recall_roi_%s" % str(cur_thresh)] = 0
        metric["recall_rcnn_%s" % str(cur_thresh)] =0

    dataset = data_loader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info("*****************EPOCH %s EVALUATION*************" % epoch_id)
    model.eval()

    progress_bar = tqdm.tqdm(total=len(data_loader),leave=True,desc="eval",dynamic_ncols=True)
    start_time =time.time()
    time_data = []
    for i,data in enumerate(data_loader):

        input_dict = example_convert_to_torch(data)
        start_infer = time.time()
        pred_dict, ret_dict = model(input_dict)
        # print(float(data_loader.batch_size))
        print("第 %d 帧用的时间:" % (i),(time.time()-start_infer)/float(data_loader.batch_size))
        time_data.append((time.time()-start_infer)/float(data_loader.batch_size))
        disp_dict = {}
        if dataset.split is not "test":
            statistic_info(ret_dict,metric,disp_dict)
        annos = dataset.generate_prediction_dicts(input_dict,pred_dict,class_names,
                                            output_path=final_output_dir)
        det_annos += annos
        progress_bar.set_postfix(disp_dict)
        progress_bar.update()
        #if i>10:
            #break# 为了测试后边的模块而加的，真实运行应该去掉
    progress_bar.close()
    TIME_FILE = "/home/liang/for_ubuntu502/PVRCNN-V1.1/output/figures"
    os.makedirs(TIME_FILE,exist_ok=True)
    TIME_FILE = os.path.join(TIME_FILE,"interfer_time_list.pth")
    with open(TIME_FILE,"wb") as f:
        pickle.dump(time_data,f)
    logger.info("*********Performance of EPOCH %s *******************" % epoch_id)
    second_per_example = (time.time()-start_time) / len(data_loader.dataset)
    logger.info("Generate label finished(sec per example: %.4f second)" % second_per_example)
    ret_dict= {}

    gt_num_cnt = metric["num_gt"]
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric["recall_roi_%s" % str(cur_thresh)]/ max(gt_num_cnt,1)
        cur_rcnn_recall = metric["recall_rcnn_%s" % str(cur_thresh)] / max(gt_num_cnt,1)
        logger.info("recall_roi_%s:%f" % (cur_thresh,cur_roi_recall))
        logger.info("recall_rcnn_%s:%f" % (cur_thresh,cur_rcnn_recall))
        ret_dict["recall_roi_%s" % str(cur_thresh)] = cur_roi_recall
        ret_dict["recall_rcnn_%s" % str(cur_thresh)] = cur_rcnn_recall
    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno["name"].__len__()
    logger.info("Average predicte number of objects(%d samples): %.3f"
                % (len(det_annos),total_pred_objects / max(1,len(det_annos))))
    reuslt_file_path = os.path.join(result_dir,"result.pkl")
    with open(reuslt_file_path,"wb") as f:
        pickle.dump(det_annos,f)
    if dataset.split is not "test":
        result_str, result_dict = dataset.evaluation(det_annos,class_names,
                                       eval_metric = "kitti")

        logger.info(result_str)
        ret_dict.update(result_dict)
    logger.info("Result is save to %s" % result_dir )
    logger.info("*********************Evaluation Done*********************")
    return ret_dict





def statistic_info(ret_dict,metric,disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric["recall_roi_%s" % str(cur_thresh)] += ret_dict["roi_%s" % str(cur_thresh)]
        metric["recall_rcnn_%s" % str(cur_thresh)] += ret_dict["rcnn_%s" % str(cur_thresh)]
        metric["num_gt"] += ret_dict["gt"]
    thresh_mid =cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[1]
    disp_dict["recall_%s" % str(thresh_mid)] = \
            "(%d,%d) / %d" % (metric["recall_roi_%s" % str(thresh_mid)],
                              metric["recall_rcnn_%s" % str(thresh_mid)],
                              metric["num_gt"])




