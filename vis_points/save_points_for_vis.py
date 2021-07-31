from pvdet.dataset.Dataset import KittiDataset
from pvdet.tools.config import cfg
import pickle
import os
from vis_points.vis_config import cfg_vis
import numpy as np

def generate_data_for_vis(split,sample_idx):
    dataset = KittiDataset(datapath=cfg.DATA_DIR,
                           class_name=cfg.CLASS_NAMES,
                           training=True,
                           split=split,
                           #split = cfg.MODEL["TRAIN" if training else "TEST"].SPLIT,
                           logger=None
                           )
    raw_dataset = KittiDataset(datapath=cfg.DATA_DIR,
                           class_name=cfg.CLASS_NAMES,
                           training=False,
                           split=split,
                           #split = cfg.MODEL["TRAIN" if training else "TEST"].SPLIT,
                           logger=None
                           )
    process_data = None
    raw_data = None

    for i,pro_data in enumerate(dataset):
        if pro_data["sample_idx"] == sample_idx:
            process_data = pro_data
            break
        if i>len(dataset):
            raise UnboundLocalError
    for i,raw_data in enumerate(raw_dataset):
        if raw_data["sample_idx"] == sample_idx:
            break
        if i>len(raw_dataset):
            raise UnboundLocalError
    assert process_data["sample_idx"] == raw_data["sample_idx"]


    raw_data_path = "/home/liang/PVRCNN-V0/debug_file"
    os.makedirs(raw_data_path, exist_ok=True)
    raw_data_path= os.path.join(raw_data_path,"raw_%s.pth" % raw_data["sample_idx"])
    pro_data_path = "/home/liang/PVRCNN-V0/debug_file"
    os.makedirs(pro_data_path, exist_ok=True)
    pro_data_path=os.path.join(pro_data_path, "process_%s.pth" % pro_data["sample_idx"])

    del raw_data["calib"]
    del pro_data["calib"]
    with open(raw_data_path,"wb") as f:
        pickle.dump(raw_data,f)

    with open(pro_data_path,"wb") as f:
        pickle.dump(pro_data,f)


def save_gt_lidar_v1():
    """
    add the cls id to the lidar box (:,7+cls_id)
    """
    gt_annos_dict_path = cfg_vis.kitti_val_info
    gt_output_dir = cfg_vis.kitti_gt_dir
    os.makedirs(gt_output_dir,exist_ok=True)
    with open(gt_annos_dict_path, "rb") as f:
        gt_annos_dict = pickle.load(f)
    index = []
    gt_class_names = []
    gt_lidar_box_list = []
    for i, det in enumerate(gt_annos_dict):
        index.append(int(det["image"]["image_idx"]))
        gt_class_names.append(det["annos"]["name"])
        gt_lidar_box_list.append(det["annos"]["gt_boxes_lidar"])
    # real_index = np.argsort(index)
    # gt_lidar_box_list = [gt_lidar_box_list[i] for i in real_index]
    # gt_class_names = [gt_class_names[i] for i in real_index]
    gt_lidar_box = dict({"gt_lidar_box_list":gt_lidar_box_list,
                    "gt_class_names":gt_class_names})
    gt_lidar_boxs_path = os.path.join(gt_output_dir, "lidar_gt_list.pth")
    with open(gt_lidar_boxs_path, "wb") as f:
        pickle.dump(gt_lidar_box, f)
    print("save success!")
if __name__ == '__main__':
    save_gt_lidar_v1()
#scp -r -P 9006 liang@ec5dbab349166515.natapp.cc:~/PVRCNN-V0/debug_file /media/liang/TOSHIBA\ EXT/vis_points/
#scp -r liang@192.168.1.12:~/PVRCNN-V0/debug_file /media/liang/TOSHIBA\ EXT/vis_points/