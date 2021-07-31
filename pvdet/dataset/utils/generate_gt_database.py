import numpy as np
import os
import torch
import concurrent.futures as futures
import pickle

from pvdet.dataset.Dataset import KittiDataset
from pvdet.dataset.roiaware_pool3d import roiaware_pool3d_utils
from pvdet.tools.config import cfg






class GnerateGtDatabase(KittiDataset):
    def __init__(self,datapath=cfg.DATA_DIR,class_name=cfg.CLASS_NAMES,training=True,split="train",num_workers=2):
        super().__init__(datapath=datapath,class_name=class_name, training= training,split=split)
        self.num_workers = num_workers
        self.datapath = datapath

    def get_all_info(self):

        sample_idx_list = self.sample_idx_list[:10]#正式运行时去掉

        with futures.ThreadPoolExecutor(self.num_workers) as executor:
            info_list = executor.map(self.get_info, sample_idx_list, [self.split] * len(sample_idx_list))
        return list(info_list)
    def create_gt_database(self,dppath=None,dbinfopath= None,use_classes=None):
        if (dppath is None) and (dbinfopath is None) :
            database_save_path = os.path.join(self.datapath,
                                          "gt_database" if self.split == "train" else ("gt_database_%s" % self.split))
            db_info_save_path = os.path.join(self.datapath, "kitti_info_%s.pkl" %  self.split)
        else:
            assert os.path.exists(dppath)
            assert os.path.exists(dbinfopath)
            database_save_path=dppath
            db_info_save_path =dbinfopath

        os.makedirs(database_save_path,exist_ok=True)
        os.makedirs(db_info_save_path, exist_ok=True)
        info_list = self.get_all_info()#获得在split下的所有样本的标注信息，
        all_db_infos = {}
        #return info_list
        for k in range(len(info_list)):
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
                filepath = os.path.join(database_save_path,filename)
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

        with open(db_info_save_path,"wb") as f:
            pickle.dump(all_db_infos,f)






if __name__=="__main__":
    dataset= GnerateGtDatabase()
    dataset.create_gt_database(dppath="/media/liang/Elements/PVRCNN-V0/database",
                               dbinfopath="/media/liang/Elements/PVRCNN-V0/database_information")











