
from single_stage_model.configs.single_stage_config import cfg


import numpy as np
from torch.utils.data import Dataset
import os
from skimage import io

# import pvdet.dataset.utils.calibration as calibration
from single_stage_model.dataset import calibration

# import pvdet.dataset.utils.object3d_utlis as object3d_utils
from single_stage_model.dataset import object3d_utils

# import pvdet.dataset.utils.box_utils as box_utils
from single_stage_model.dataset import  box_utils
import cv2



class KittiTemplate(Dataset):
    def __init__(self,datapath=cfg.DATA_DIR,
                 class_name=cfg.CLASS_NAMES,
                 training=True,split="train",
                 logger=None,
                 train_all=False,):
        super().__init__()
        self.root_path=datapath
        self.root_split_path = os.path.join(self.root_path, "training" if split != "test" else "testing")
        self.split = split
        if split in ["train","val","test","train_80%","val_20%"]:
            split_dir = os.path.join(self.root_path,"ImageSets",split+".txt")
            if train_all:
                split_dir_val = os.path.join(self.root_path,"ImageSets","val.txt")

        self.sample_idx_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None
        if train_all:
            self.sample_idx_list += [x.strip() for x in open(split_dir_val).readlines()] if os.path.exists(split_dir_val) else None
        self.training = training
        self.class_names = class_name
        self.mode = "TRAIN" if self.training else "TEST"

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.root_split_path, 'velodyne', '%s.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_image_shape(self, idx):
        img_file = os.path.join(self.root_split_path, 'image_2', '%s.png' % idx)
        assert os.path.exists(img_file)
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_img(self,idx):
        image_file = os.path.join(self.root_split_path,"image_2","%s.png"%idx)
        return cv2.imread(image_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.root_split_path, 'calib', '%s.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.root_split_path, 'label_2', '%s.txt' % idx)
        assert os.path.exists(label_file)
        return object3d_utils.get_objects_from_label(label_file)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.root_split_path, 'planes', '%s.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def get_info(self,sample_idx,has_label=True,count_inside_pts=True):
        print('%s sample_idx: %s' % (self.split, sample_idx))
        info = {}
        pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
        info['point_cloud'] = pc_info

        image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
        info['image'] = image_info
        calib = self.get_calib(sample_idx)

        P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
        R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
        R0_4x4[3, 3] = 1.
        R0_4x4[:3, :3] = calib.R0
        V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
        calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

        info['calib'] = calib_info

        if has_label:
            obj_list = self.get_label(sample_idx)
            annotations = {}
            annotations['name'] = np.array([obj.cls_type for obj in obj_list])
            annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
            annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
            annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
            annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
            annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
            annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
            annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
            annotations['score'] = np.array([obj.score for obj in obj_list])
            annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

            num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
            num_gt = len(annotations['name'])
            index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
            annotations['index'] = np.array(index, dtype=np.int32)

            loc = annotations['location'][:num_objects]
            dims = annotations['dimensions'][:num_objects]
            rots = annotations['rotation_y'][:num_objects]
            loc_lidar = calib.rect_to_lidar(loc)
            l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
            gt_boxes_lidar = np.concatenate([loc_lidar, w, l, h, rots[..., np.newaxis]], axis=1)
            annotations['gt_boxes_lidar'] = gt_boxes_lidar

            info['annos'] = annotations

            if count_inside_pts:
                points = self.get_lidar(sample_idx)
                calib = self.get_calib(sample_idx)
                pts_rect = calib.lidar_to_rect(points[:, 0:3])

                fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                pts_fov = points[fov_flag]
                corners_lidar = box_utils.boxes3d_to_corners3d_lidar(gt_boxes_lidar)
                num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                for k in range(num_objects):
                    flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                    num_points_in_gt[k] = flag.sum()
                annotations['num_points_in_gt'] = num_points_in_gt

        return info


    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                gt_boxes_lidar = np.concatenate([loc_lidar, w, l, h, rots[..., np.newaxis]], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes3d_to_corners3d_lidar(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        # temp = process_single_scene(self.sample_id_list[0])
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        '''
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param img_shape:
        :return:
        '''
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag