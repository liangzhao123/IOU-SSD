import numpy as np
import os
import torch
from vis_points.vis_config import cfg_vis
from pvdet.dataset.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
import mayavi.mlab as mlab
import vis_points.utils as utils
from vis_points.fps_utils import fps_func_utils
from torch import nn
import pvdet.dataset.utils.object3d_utlis as object3d_utils
import pvdet.dataset.utils.calibration as calibration
from pvdet.dataset.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from pvdet.dataset.utils.box_utils import boxes3d_to_corners3d_lidar
import time

def get_lidar(idx,root_path,split):
    # split can chose training or testing
    assert os.path.exists(root_path)
    lidar_file = os.path.join(root_path,split, 'velodyne', '%s.bin' % idx)
    return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

def get_calib(calib_path):
    assert os.path.exists(calib_path)
    return calibration.Calibration(calib_path)

def get_labels(idx,root_path,split):
    label_path = os.path.join(root_path, split, 'label_2', '%s.txt' % idx)
    assert os.path.exists(label_path)
    obj_list =  object3d_utils.get_objects_from_label(label_path)
    calib_path = os.path.join(root_path, split,"calib",'%s.txt' % idx)
    calib = get_calib(calib_path)
    annotations = {}
    annotations["name"] = np.array([obj.cls_type for obj in obj_list])
    annotations["truncated"] = np.array([obj.truncation for obj in obj_list])
    annotations["occluded"] = np.array([obj.occlusion for obj in obj_list])
    annotations["alpha"] = np.array([obj.alpha for obj in obj_list])
    annotations["bbox"] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
    annotations["dimensions"] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])
    annotations["location"] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
    annotations["rotation_y"] = np.array([obj.ry for obj in obj_list])
    annotations["score"] = np.array([obj.score for obj in obj_list])
    annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

    num_object = len([obj.cls_type for obj in obj_list if obj.cls_type != "DontCare"])
    num_gt = len(annotations["name"])
    index = list(range(num_object)) + [-1] * (num_gt - num_object)
    annotations["index"] = np.array(index, dtype=np.int32)

    loc = annotations["location"][:num_object]
    dims = annotations["dimensions"][:num_object]
    rots = annotations["rotation_y"][:num_object]
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar = calib.rect_to_lidar(loc)
    gt_boxes_lidar = np.concatenate([loc_lidar, w, l, h, rots[..., np.newaxis]], axis=1)
    return gt_boxes_lidar

def fps_d(points,npoints):
    batch_size = points.shape[0]
    points_xyz = points[:,:,0:3]
    points_xyz = torch.from_numpy(points_xyz).cuda().type(torch.float32)
    start = time.time()
    keypoints_indx = fps_func_utils.fps_with_distance(points_xyz,npoints)
    print(time.time()-start)
    keypoints_indx = keypoints_indx.cpu().numpy()
    keypoinst_list = []
    for bs_idx in range(batch_size):
        cur_idx = keypoints_indx[bs_idx]
        keypoinst_list.append(points[bs_idx,cur_idx])
    keypoinst = np.concatenate(keypoinst_list, axis=0)
    return keypoinst

def random_select_half(idx,ratio=0.8):
    low = 0
    high = idx.numel()
    size = int(idx.numel()*ratio)
    selected_half_idx = torch.randint(low=low,high=high,size=(size,)).long().cuda()
    idx = idx[selected_half_idx]
    return idx


def fps_f(points,seg_idx,npoints):
    # points (B,N,4) ndarray
    # features (B,N,C2) ndarray
    # return keypoinst (B,num of keypoints,4)
    batch_size = points.shape[0]
    points_xyz = points[:,:,0:3]
    # pts_f = np.concatenate([points_xyz,features],axis=-1)
    seg_idx = seg_idx.reshape(-1)
    seg_idx_neg =(seg_idx==0).nonzero()
    seg_idx_pos = seg_idx.nonzero()
    if (seg_idx_pos.numel()>0):
        seg_idx_pos = random_select_half(seg_idx_pos,ratio=0.2)
        seg_idx_neg = random_select_half(seg_idx_neg,ratio=0.8)
        index_fusion = torch.cat([seg_idx_pos,seg_idx_neg],dim=0)
    else:
        seg_idx_neg = random_select_half(seg_idx_neg)
        index_fusion = seg_idx_neg
    seg_idx = torch.tensor(index_fusion,dtype=torch.int32).cuda()
    start = time.time()
    keypoints_indx = fps_func_utils.fps_with_featres(points_xyz.contiguous(),seg_idx,npoints)
    print(time.time()-start)
    keypoints_indx = keypoints_indx.to("cpu").numpy()
    points = points.cpu().numpy()
    keypoinst_list= []
    for bs_idx in range(batch_size):
        cur_idx = keypoints_indx[bs_idx]
        keypoinst_list.append(points[bs_idx,cur_idx])
    keypoints = np.concatenate(keypoinst_list,axis=0)
    return keypoints



class model(nn.Module):
    def __init__(self,data):
        # data is the point cloud (B,N,4)
        super().__init__()
        self.data = data
    def forward(self,):
        B,N,_ = self.data.shape()
        segment_labels = torch.cuda.FloatTensor(B,N)
        points_in_boxes_gpu(self.data)
        pass

def save_forgroud_point_idx():
    data_root_path = cfg_vis.DATA_DIR
    split = "training"
    sample_id = "000211"
    lidar_points = get_lidar(sample_id,data_root_path,split)
    gt_lidar_box = get_labels(sample_id,data_root_path,split)
    seg_features = points_in_boxes_cpu(torch.from_numpy(lidar_points[:, :3]),
                                       torch.from_numpy(gt_lidar_box[:, :7]))
    seg_features = torch.tensor(seg_features.clone().detach(),dtype=torch.int32)
    seg_features = seg_features.sum(dim = 0)  # segment labels 0 or 1
    seg_features = torch.where(seg_features > torch.tensor(0), torch.tensor(1), torch.tensor(0))
    idx = seg_features.nonzero().numpy()
    idx = np.reshape(idx,(-1))
    idx_path = os.path.join(data_root_path,"seg_label",split)

    os.makedirs(idx_path,exist_ok=True)
    idx_file = os.path.join(idx_path,"%s.bin" % sample_id)
    idx.tofile(idx_file)



def main1():
    data_root_path = cfg_vis.DATA_DIR
    lidar_points = get_lidar("000211",data_root_path,"training")
    gt_lidar_box = get_labels("000211",data_root_path,"training")
    # features= model(lidar_points)
    plot_str = "fps_f"
    if plot_str== "fps_f":
        seg_features = points_in_boxes_cpu(torch.from_numpy(lidar_points[:, :3]), torch.from_numpy(gt_lidar_box[:, :7])).numpy()
        seg_features = seg_features.sum(axis = 0) # segment labels 0 or 1
        seg_features = np.where(seg_features>0,1,0)
        seg_features = seg_features[np.newaxis,...,np.newaxis]
        lidar_points = lidar_points[np.newaxis,...]
        seg_features = torch.from_numpy(seg_features).type(torch.int32).cuda()
        lidar_points = torch.from_numpy(lidar_points).cuda()
        keypoints = fps_f(lidar_points, seg_features,1024)
    elif plot_str== "fps_d":
        lidar_points = lidar_points[np.newaxis,...]
        keypoints = fps_d(lidar_points,1024)
    else:
        assert NotImplementedError
    points_in_boxes , points_out_boxes = utils.points_indices(keypoints, gt_lidar_box)
    points_size = np.array([3]).repeat(points_in_boxes.shape[0], 0)
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
    )
    fig = utils.draw_lidar(points_out_boxes, fig=fig)
    gt_boxes_corners = boxes3d_to_corners3d_lidar(gt_lidar_box[:, :7])
    fig = utils.draw_gt_boxes3d(gt_boxes_corners, color=(1, 0, 0), fig=fig)
    mlab.points3d(points_in_boxes[:, 0],
                  points_in_boxes[:, 1],
                  points_in_boxes[:, 2],
                  points_size,
                  color=(1, 0, 0),
                  mode="sphere",
                  scale_factor=0.1,
                  figure=fig)
    mlab.show()



if __name__ == '__main__':
    # save_forgroud_point_idx()
    main1()
