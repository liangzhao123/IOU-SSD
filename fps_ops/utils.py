import torch
import time
from vis_points.fps_utils import fps_func_utils
import numpy as np
from new_train.config import cfg
def random_select_half(idx,ratio=1):
    low = 0
    high = idx.numel()
    size = int(idx.numel()*ratio)
    selected_half_idx = torch.randint(low=low,high=high,size=(size,)).long().cuda()
    idx = idx[selected_half_idx]
    return idx

def fps_f_v1(points,seg_score,npoints,fg_score=cfg.MODEL.Seg_net.fg_threshold):
    """
    args:points:(B,N,3)
        seg_score:(N)
        npoints:(1),equals K
    return keypoints:(K,3)
    """
    start = time.time()
    fg_idx = torch.nonzero(seg_score > fg_score).view(-1)
    points_xyz = points[:,:,0:3]
    bg_idx = torch.nonzero(seg_score <= fg_score).view(-1)
    if cfg.print_info:
        print("find fg and bg in fps spend time ", time.time() - start)
    if (fg_idx.numel()>0):
        start = time.time()
        fg_idx = random_select_half(fg_idx,ratio=0.2)
        bg_idx = random_select_half(bg_idx,ratio=0.8)
        index_fusion = torch.cat([fg_idx,bg_idx],dim=0)
        if cfg.print_info:
            print("select fg and bg in fps spend time ",time.time()-start)
    else:
        index_fusion = random_select_half(bg_idx,ratio = 0.5)
    start = time.time()
    seg_idx = torch.tensor(index_fusion,dtype=torch.int32).cuda()
    if cfg.print_info:
        print("copy to cuda in fps spend time ", time.time() - start)
    start = time.time()
    keypoints_indx = fps_func_utils.fps_with_featres(points_xyz.contiguous(),seg_idx,npoints).long()
    if cfg.print_info:
        print("fps_func_utils.fps_with_featres spend time",time.time()-start)
    start = time.time()

    keypoints = points[0,keypoints_indx[0]]
    if cfg.print_info:
        print("indices keypoints according indx spend time",time.time()-start)

    return keypoints

def fps_f_v2(points,seg_score,npoints):
    seg_score_max,seg_indx = torch.max(seg_score,dim=1)
    seg_fg_indx = seg_indx.nonzero()
    seg_bg_indx = (seg_indx==0).nonzero()
    if (seg_fg_indx.numel() > 0):
        seg_fg_indx = random_select_half(seg_fg_indx, ratio=0.2)
        seg_bg_indx = random_select_half(seg_bg_indx, ratio=0.8)
        index_fusion = torch.cat([seg_fg_indx, seg_bg_indx], dim=0)
    else:
        index_fusion = seg_bg_indx
    seg_indx = torch.tensor(index_fusion, dtype=torch.int32).cuda()
    keypoints_indx = fps_func_utils.fps_with_featres(points.contiguous(), seg_indx, npoints).long()
    keypoints = points[0, keypoints_indx[0]]
    return keypoints

def fps_f(points,seg_idx,npoints):
    """
    args:points:(B,N,3)
        seg_idx:(N)
        npoints:(1),equals K
    return keypoints:(K,3)
    """
    # points (B,N,4) ndarray
    # features (B,N,C2) ndarray
    # return keypoinst (B,num of keypoints,4)
    # points = points.unsqueeze(dim=0)
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
    keypoints_indx = fps_func_utils.fps_with_featres(points_xyz.contiguous(),seg_idx,npoints).long()
    # print("fps_func_utils.fps_with_featres spend time",time.time()-start)
    start = time.time()
    keypoints = points[0,keypoints_indx[0]]
    # keypoints_indx = keypoints_indx.to("cpu").numpy()
    # points = points.cpu().numpy()
    # keypoinst_list= []
    # cur_idx = keypoints_indx[0]
    # keypoinst_list.append(points[0,cur_idx])
    # keypoints = np.concatenate(keypoinst_list,axis=0)
    # print("dtat convert in fps_f spend time",time.time()-start)

    return keypoints

if __name__ == '__main__':
    a = torch.tensor([1,2,3],dtype=torch.float32).cuda()
    indx = torch.tensor([0,2],dtype=torch.int64).cuda()
    b = a[indx]
    print(b)