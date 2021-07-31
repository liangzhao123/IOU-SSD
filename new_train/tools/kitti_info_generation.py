import numpy as np
from new_train.config import cfg
import os
import torch
import concurrent.futures as futures
from new_train.dataset.kitti import KittiTemplate
from pvdet.dataset.roiaware_pool3d import roiaware_pool3d_utils
import pickle





class GnerateGtDatabase(KittiTemplate):
    def __init__(self,datapath=cfg.DATA_DIR,
                 class_name=cfg.CLASS_NAMES,
                 num_workers=4,
                 training=True,
                 split="train",
                 logger=None,
                 train_all= False):
        super().__init__(datapath=datapath,class_name=class_name, training= training,split=split,train_all=train_all)
        self.num_workers = num_workers
        self.train_all = train_all
    def set_train_all(self,train_all):
        self.train_all = train_all

    def set_split(self,split):
        self.split = split
        if split in ["train","val","test"]:
            split_dir = os.path.join(self.root_path,"ImageSets",split+".txt")
        self.sample_idx_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None
        self.root_split_path = os.path.join(self.root_path, "training" if split != "test" else "testing")

    def get_all_info(self):

        sample_idx_list = self.sample_idx_list
        self.num_workers = 15

        with futures.ThreadPoolExecutor(self.num_workers) as executor:
            info_list = executor.map(self.get_info, sample_idx_list, [True if self.split !="test" else False] * len(sample_idx_list))
        print("kitti info number :%d"%len(sample_idx_list))
        return list(info_list)


    def greate_kitti_info(self,info_path =cfg.kitti_info_path,split = None):
        os.makedirs(info_path,exist_ok=True)
        if split == "trainval":
            kitti_info_train_path = os.path.join(info_path, "kitti_infos_train.pkl" )
            kitti_info_val_path = os.path.join(info_path, "kitti_infos_val.pkl")
            with open(kitti_info_train_path,"rb") as f:
                train_info_list = pickle.load(f)
            with open(kitti_info_val_path,"rb") as f:
                val_info_list = pickle.load(f)
            kitti_info_trainval_path = os.path.join(info_path, "kitti_infos_trainval.pkl" )
            train_val_info_list = train_info_list + val_info_list
            with open(kitti_info_trainval_path,"wb") as f:
                pickle.dump(train_val_info_list,f)
            print("save  train_val info file to %s" % kitti_info_trainval_path)
            return 0
        # elif split in ["train_80%","val_20%"]:
        #     kitti_info_train_path = os.path.join(info_path, "kitti_infos_%s.pkl" % split)


        kitti_info_path = os.path.join(info_path, "kitti_infos_%s.pkl" % split)
        info_list = self.get_all_info()
        with open(kitti_info_path, "wb") as f:
            pickle.dump(info_list, f)
        print("save  %s info file to %s" % (split,kitti_info_path))
        return 0


    def create_gt_database(self,kitti_info_path,database_info_output_path,
                           database_output_path= None,use_classes=None,split=None):
        info_file = os.path.join(kitti_info_path,"kitti_infos_%s.pkl" % self.split)
        if self.train_all:
            info_file = os.path.join(kitti_info_path,"kitti_infos_trainval.pkl" )
        with open(info_file,"rb") as f:
            info_list = pickle.load(f)
        if split not in ["train","train_80%","trainval"]:
            print("jump ground truth generate")
            return 0

        all_db_infos = {}
        #return info_list
        for k in range(len(info_list)):
            assert split in ["train","train_80%","trainval"]
            print("gt_database_sample %d/%d" % (k+1, len(info_list)))
            info = info_list[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes_lidar = annos['gt_boxes_lidar']

            num_obj = gt_boxes_lidar.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:,:3]),torch.from_numpy(gt_boxes_lidar)).numpy()#返回值是[num_box,npoints]在box里是1否则为0

            for j in range(num_obj):
                filename = "%s_%s_%d.bin" %(sample_idx, names[j],j)
                os.makedirs(database_output_path,exist_ok=True)
                filepath = os.path.join(database_output_path,filename)
                gt_points = points[point_indices[j]>0]

                gt_points[:,:3] -=gt_boxes_lidar[j,:3]#计算出在box内的点到box中心的坐标差
                with open(filepath, 'w') as f:
                    gt_points.tofile(filepath)

                if (use_classes is None) or (names[j] in use_classes):
                    databse_path = str(filepath)
                    db_info = {"name":names[j],
                               "path":databse_path,
                               "image_idx":sample_idx,
                               "gt_idx":j,
                               "box3d_lidar":gt_boxes_lidar[j],
                               "num_points_in_gt":gt_points.shape[0],
                               "difficulty":difficulty[j],
                               "bbox":bbox[j],
                               "score":annos["score"][j]}
                    if names[j] in all_db_infos:
                        all_db_infos[names[j]].append(db_info)
                    else:
                        all_db_infos[names[j]]=[db_info]

        for key ,value in all_db_infos.items():
            print("Dataset %s: %d"% (key,len(value)))


        db_info_file_path = os.path.join(database_info_output_path,"kitti_dbinfos_%s.pkl" % self.split)
        if self.train_all:
            db_info_file_path = os.path.join(database_info_output_path,"kitti_dbinfos_trainval.pkl")
        with open(db_info_file_path,"wb") as f:
            pickle.dump(all_db_infos,f)

def get_train_80():
    file_type = ["train_80%", "val_20%", ]
    for i, split in enumerate(file_type):
        dataset = GnerateGtDatabase(split=file_type[i])
        dataset.greate_kitti_info(cfg.KITTI_INFO_PATH, split=split)
    print("done")


def generate_db_train_80():
    file_type = ["train_80%"]
    for i, split in enumerate(file_type):
        dataset = GnerateGtDatabase(split=file_type[i])
        dataset.create_gt_database(kitti_info_path=cfg.KITTI_INFO_PATH,
                                   database_info_output_path=cfg.KITTI_INFO_PATH,
                                   database_output_path=cfg.GT_DATABASE)


def create_v1():
    file_type = ["train", "val", "trainval", "test"]
    dataset = GnerateGtDatabase(split=file_type[0])
    for split in file_type:
        if split is not "trainval":
            dataset.set_split(split)  # 这个产生的是，imageset里的train那个文件夹里的下的info，还可以使用val，
        if split is "trainval":
            dataset.set_train_all(True)
        else:
            dataset.set_train_all(False)
        dataset.greate_kitti_info(cfg.kitti_info_path, split=split)
        dataset.create_gt_database(kitti_info_path=cfg.kitti_info_path,
                                   database_info_output_path=cfg.kitti_info_path,
                                   database_output_path=cfg.gt_database,
                                   split=split)
    # train_all = True  # create all db in training fold
    # if train_all:
    #     dataset = GnerateGtDatabase(split="train", train_all=train_all)
    #     dataset.create_gt_database(kitti_info_path=cfg.kitti_info_path,
    #                                database_info_output_path=cfg.kitti_info_path,
    #                                database_output_path=cfg.gt_database)

if __name__=="__main__":
    create_v1()
    print("done")











