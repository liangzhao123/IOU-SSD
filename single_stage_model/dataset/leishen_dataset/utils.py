import numpy as np
import glob
import os

import pickle


DATA_DIR = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/rosbag/leishen_e70_32/dataset_image_pcd"
OUTPUT_DIR = "/home/liang/for_ubuntu502/PVRCNN-V1.1/output"
CODE_DIR =  "/home/liang/for_ubuntu502/PVRCNN-V1.1"
def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object(line) for line in lines]
    return objects

def get_info(objects,anno=False,drop_bus=False):
    annotations = {}
    gt_boxes_list = []
    for object in objects:
        if anno:
            gt_box = [*object.loc,object.w,object.l,object.h,-object.ry,object.cls_id]
        else:
            gt_box = [*object.loc, object.w, object.l, object.h, object.ry, object.cls_id]
        if drop_bus and object.cls_type=="Bus":
            continue
        gt_boxes_list.append(gt_box)
    return np.array(gt_boxes_list)


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Rider': 3, 'Bus': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

def cls_id_to_type(cls_id):
    id_to_type = {1: "Car", 2:'Pedestrian', 3:'Rider', 4:'Bus'}
    if cls_id not in id_to_type.keys():
        return -1
    return id_to_type[cls_id]

class Object(object):
    def __init__(self,line):
        super().__init__()
        label = line.strip().split(' ')
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.w = float(label[4])
        self.l = float(label[5])
        self.h = float(label[6])
        self.ry = -float(label[7])
        self.loc = np.array((float(label[1]), float(label[2]), float(label[3])-self.h/2.0), dtype=np.float32)

def read_bin(path):
    points  = np.fromfile(path,dtype=np.float32).reshape(-1,4)
    return points



def center2corner_leishen(boxes3d, bottom_center=True):
    """
        :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords, see the definition of ry in KITTI dataset
        :param z_bottom: whether z is on the bottom center of object
        :return: corners3d: (N, 8, 3)
            7 -------- 4
           /|         /|
          6 -------- 5 .
          | |        | |
          . 3 -------- 0
          |/         |/
          2 -------- 1
        """
    boxes_num = boxes3d.shape[0]

    w, l, h = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    y_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    x_corners= np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.], dtype=np.float32).T
    if bottom_center:
        z_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        z_corners[:, 4:8] = h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        z_corners = np.array([-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), -np.sin(ry), zeros],
                         [np.sin(ry), np.cos(ry), zeros],
                         [zeros, zeros, ones]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)


    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    rotated = False
    if rotated:
        x_corners, y_corners, z_corners = temp_corners[:, :, 0], temp_corners[:, :, 1], temp_corners[:, :, 2]
    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32),temp_corners.astype(np.float32)



def get_leishen_annotation_info(path):
    anno_path_list = glob.glob(os.path.join(path,"*.txt"))
    dataset_info = os.path.join(os.path.abspath(path+"/.."),"annotation_info.txt")
    dataset_info_pth = os.path.join(os.path.abspath(path + "/.."), "annotation_info.pth")
    #for debug
    # with open(dataset_info_pth,"rb") as f:
    #     a = pickle.load(f)
    anno_path_list.sort()
    num_box = 0
    name_dict = {}
    labels_list = []
    gt_boxes_list = []
    for i in range(1,len(anno_path_list)+1):
        filename = os.path.join(path,str(i).zfill(6)+".txt")
        try :
            os.path.exists(filename)
            objects = get_objects_from_label(filename)
            labels = [i.cls_type for i in objects]
            labels_list.extend(labels)
            num_box += len(objects)
            gt_boxes = get_info(objects)
            gt_boxes_list.extend(gt_boxes)
            for k in range(len(gt_boxes)):
                gt_box = gt_boxes[k]
                if gt_box[0]>100 or gt_box[1]>100 or gt_box[2]>100 or gt_box[0]<-100 or gt_box[1]<-100 or gt_box[2]<-100:
                    print("error annotation in :",filename)
        except:
            print(filename)
    loc_np = np.array(gt_boxes_list,dtype=np.float32)[:,0:3]
    loc_max = [loc_np[:,0].max(),loc_np[:,1].max(),loc_np[:,2].max()]
    loc_min = [loc_np[:,0].min(),loc_np[:,1].min(),loc_np[:,2].min()]

    labels_unique = np.unique(labels_list)
    label_dict = dict(zip(labels_unique,[0]*len(labels_unique)))
    for i in labels_list:
        label_dict[i] += 1
    box_size_dict = dict(zip(labels_unique, [0] * len(labels_unique)))

    for i in range(len(gt_boxes_list)):
        gt_box = gt_boxes_list[i]
        class_name = cls_id_to_type(gt_box[7])
        if box_size_dict[class_name]==0:
            box_size_dict[class_name] = []
        box_size_dict[class_name].append(gt_box[3:6])
    mean_size_dict = {}
    for i in box_size_dict.keys():
        mean_size = np.array(box_size_dict[i]).mean(axis=0)
        mean_size_dict[i] = list(mean_size)# w,l,h
    info = {}
    info["loc_max"] = loc_max
    info["loc_min"] = loc_min
    info["number_of_box"] = label_dict
    info["average_box_dimension"] = mean_size_dict
    info["total_box"] = num_box
    data_f = open(dataset_info,"w")
    with open(dataset_info_pth,"wb") as f:
        pickle.dump(info,f)
    print("save leishen annotation information to:",dataset_info_pth)
    data_f.write(str(info))
    data_f.close()
    print(info)



if __name__ == '__main__':
    pass
    # pass

