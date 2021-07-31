import numpy as np
import glob
import os
class SplitData:
    def __init__(self,datapath,train_ratio=0.8):
        self.train_ratio = train_ratio
        self.data_dir = datapath
        self.bin_dir = os.path.join(self.data_dir,"bin")
        self.get_info()
        self.train_path = os.path.join(self.data_dir,"train.txt")
        self.val_path = os.path.join(self.data_dir, "val.txt")
    def get_info(self):
        self.filename_list = glob.glob(os.path.join(self.bin_dir,"*.bin"))
        print("Dataset number:",len(self.filename_list))
        self.filename_list.sort()
    def split(self,):
        num_train = int(len(self.filename_list) * self.train_ratio)
        num_val = int(len(self.filename_list)) - num_train
        step = int(np.round(num_train/num_val,0))
        f_train = open(self.train_path,"w")
        f_val = open(self.val_path,"w")
        count = 0
        num_train_ = 0
        num_val_ = 0
        for i in range(0,len(self.filename_list)):
            item = self.filename_list[i].split("/")[-1]
            item = item.split(".")[0]
            if count>=step:
                f_val.write(item+"\n")
                count = 0
                num_val_ +=1
            else:
                f_train.write(item+"\n")
                count+=1
                num_train_ += 1
        f_train.close()
        f_val.close()
        print("save %d train to %s" %(num_train_,self.train_path))
        print("save %d val to %s" % (num_val_, self.val_path))


if __name__ == '__main__':
    data_dir = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/rosbag/leishen_e70_32/dataset_image_pcd"
    spliter = SplitData(data_dir)
    spliter.split()