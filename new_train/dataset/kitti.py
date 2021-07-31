from new_train.config import cfg



import numpy as np
from torch.utils.data import Dataset
import os
from skimage import io
import pvdet.dataset.utils.calibration as calibration
import pvdet.dataset.utils.object3d_utlis as object3d_utils
import pvdet.dataset.utils.box_utils as box_utils

import cv2
import pvdet.dataset.utils.common_utils as common_utils
import pickle
from pvdet.dataset.data_augmentation.dbsampler import DataBaseSampler

from pvdet.dataset.data_augmentation import augmentation_utils
import spconv
from collections import defaultdict
import copy

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


class KittiDataset(KittiTemplate):
    def __init__(self,datapath=cfg.DATA_DIR,class_name=cfg.CLASS_NAMES,training=True,split="train",logger=None,args=None,train_all=False):
        super().__init__(datapath=datapath,
                 class_name=class_name,
                 training=training,split=split,
                 logger=logger,
                    train_all=train_all)

        self.train_all = train_all
        if os.path.exists(cfg.DATA_CONFIG.AUGMENTATION.DB_SAMPLER.DB_INFO_PATH[0]):
            self.db_sampler_init(class_name, logger)
        self.voxel_generator_init()
        self.point_cloud_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
        range = np.array(np.array(self.point_cloud_range[3:],dtype=np.float32) - np.array(self.point_cloud_range[:3],dtype=np.float32))
        self.voxel_size = self.voxel_generator.voxel_size
        self.grid_size =np.round(range/self.voxel_size).astype(np.int32)
        self.num_point_features =cfg.DATA_CONFIG.NUM_POINT_FEATURES.use
        self.kitti_infos = []
        self.include_kitti_data(self.mode,logger,args)

    def include_kitti_data(self,mode,logger,args):
        if args.local_rank == 0 and logger is not None:
            logger.info("Loading Kitti Dataset")
        kitti_infos =[]
        info_paths = []
        if self.split=="train":
            info_paths = cfg.DATA_CONFIG.TRAIN.INFO_PATH
        elif self.split == "val":
            info_paths = cfg.DATA_CONFIG.TEST.INFO_PATH
        elif self.split == "test":
            info_paths = cfg.DATA_CONFIG.OFFICE_TEST_INFO_PATH
        elif self.split == "train_80%":
            info_paths = cfg.DATA_CONFIG.TRAIN.INFO_PATH_TRAIN_80
        elif self.split == "val_20%":
            info_paths = cfg.DATA_CONFIG.TRAIN.INFO_PATH_VAL_20
        else:
            raise NotImplementedError
        for info_path in info_paths:
            if os.path.exists(info_path):
                with open(info_path,"rb") as f:
                    infos = pickle.load(f)
                    kitti_infos.extend(infos)
            else:
                print("There is no such file in %s"%info_path)
                raise NotADirectoryError
        """new_kitti_infos = []
        for i,info in enumerate(kitti_infos):
            info["annos"] = filter_by_difficulty(info["annos"],cfg.DATA_CONFIG.AUTMENTATION.DB_SAMPLER.PREPARE.filter_by_difficulty)
            info["annos"] = filter_by_min_points(info["annos"],
                                           cfg.DATA_CONFIG.AUTMENTATION.DB_SAMPLER.PREPARE.filter_by_min_points)
            new_kitti_infos.extend(info)"""
        self.kitti_infos.extend(kitti_infos)

        if args.local_rank == 0 and logger is not None:
            logger.info("Total samples for kitti dataset :%d"%(len(kitti_infos)))

    def db_sampler_init(self,class_name,logger):
        self.db_sampler = None
        db_sampler_cfg =cfg.DATA_CONFIG.AUGMENTATION.DB_SAMPLER
        if self.training and db_sampler_cfg.ENABLE:
            db_infos =[]
            db_info_path = db_sampler_cfg.DB_INFO_PATH[0]
            if self.train_all:
                db_info_path = db_sampler_cfg.DB_INFO_PATH[1]
            elif self.split  == "train_80%":
                db_info_path = db_sampler_cfg.DB_INFO_PATH[2]
            with open(db_info_path,"rb") as f:
                infos=pickle.load(f)
                if db_infos.__len__()==0:
                    db_infos=infos
                else:
                    [db_infos[cls].extend(infos) for cls in db_infos.keys()]
            self.db_sampler = DataBaseSampler(
                db_infos=db_infos,sampler_cfg=db_sampler_cfg,class_names=class_name,logger=logger)


    def voxel_generator_init(self):
        voxel_generator_cfg = cfg.DATA_CONFIG.VOXEL_GENERATOR
        points = np.zeros((1,4))
        self.voxel_generator = spconv.utils.VoxelGeneratorV2(
            voxel_size=voxel_generator_cfg.VOXEL_SIZE,
            point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
            max_num_points=voxel_generator_cfg.MAX_POINTS_PER_VOXEL,
            max_voxels= cfg.DATA_CONFIG[self.mode].MAX_NUMBER_OF_VOXELS
        )
        voxel_grid = self.voxel_generator.generate(points)
        """voxel_grid:是个字典包含以下字段
        voxel：(num_voxel,point_per_voxel,xyz intensity)如(1,5,3+1)
        coordinates:(num_voxel,3)
        num_points_per_voxel:(num_voxel,)
        voxel_point_mask:(num_voxel,points_per_voxel,1)若体素内的点小于max_point_per_voxel则会自动补齐，但是mask为0
        """



    def prepare_data(self, input_dict, has_label=True):
        """
        :param input_dict:
            sample_idx: string
            calib: object, calibration related
            points: (N, 3 + C1)
            gt_boxes_lidar: optional, (N, 7) [x, y, z, w, l, h, rz] in LiDAR coordinate, z is the bottom center
            gt_names: optional, (N), string
        :param has_label: bool
        :return:
            voxels: (N, max_points_of_each_voxel, 3 + C2), float
            num_points: (N), int
            coordinates: (N, 3), [idx_z, idx_y, idx_x]
            num_voxels: (N)
            voxel_centers: (N, 3)
            calib: object
            gt_boxes: (N, 8), [x, y, z, w, l, h, rz, gt_classes] in LiDAR coordinate, z is the bottom center
            points: (M, 3 + C)
        """
        sample_idx = input_dict['sample_idx']
        points = input_dict['points']
        calib = input_dict['calib']

        if has_label:
            gt_boxes = input_dict['gt_boxes_lidar'].copy()
            gt_names = input_dict['gt_names'].copy()

        if self.training:
            selected = common_utils.drop_arrays_by_name(gt_names, ['DontCare', 'Sign'])
            gt_boxes = gt_boxes[selected]
            gt_names = gt_names[selected]
            gt_boxes_mask = np.array([n in self.class_names for n in gt_names], dtype=np.bool_)

            if self.db_sampler is not None:
                road_planes = self.get_road_plane(sample_idx) \
                    if cfg.DATA_CONFIG.AUGMENTATION.DB_SAMPLER.USE_ROAD_PLANE else None
                sampled_dict = self.db_sampler.sample_all(
                    self.root_path, gt_boxes, gt_names, road_planes=road_planes,
                    num_point_features=cfg.DATA_CONFIG.NUM_POINT_FEATURES['total'], calib=calib
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict['gt_names']
                    sampled_gt_boxes = sampled_dict['gt_boxes']
                    sampled_points = sampled_dict['points']
                    sampled_gt_masks = sampled_dict['gt_masks']

                    gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
                    gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes])
                    gt_boxes_mask = np.concatenate([gt_boxes_mask, sampled_gt_masks], axis=0)

                    points = box_utils.remove_points_in_boxes3d_v0(points, sampled_gt_boxes)
                    points = np.concatenate([sampled_points, points], axis=0)

            noise_per_object_cfg = cfg.DATA_CONFIG.AUGMENTATION.NOISE_PER_OBJECT
            if noise_per_object_cfg.ENABLED:
                gt_boxes, points = \
                    augmentation_utils.noise_per_object_v3_(
                    gt_boxes,
                    points,
                    gt_boxes_mask,
                    rotation_perturb=noise_per_object_cfg.GT_ROT_UNIFORM_NOISE,
                    center_noise_std=noise_per_object_cfg.GT_LOC_NOISE_STD,
                    num_try=100
                )

            gt_boxes = gt_boxes[gt_boxes_mask]
            gt_names = gt_names[gt_boxes_mask]

            gt_classes = np.array([self.class_names.index(n) + 1 for n in gt_names], dtype=np.int32)

            noise_global_scene = cfg.DATA_CONFIG.AUGMENTATION.NOISE_GLOBAL_SCENE
            if noise_global_scene.ENABLED:
                gt_boxes, points = augmentation_utils.random_flip(gt_boxes, points)
                gt_boxes, points = augmentation_utils.global_rotation(
                    gt_boxes, points, rotation=noise_global_scene.GLOBAL_ROT_UNIFORM_NOISE
                )
                gt_boxes, points = augmentation_utils.global_scaling(
                    gt_boxes, points, *noise_global_scene.GLOBAL_SCALING_UNIFORM_NOISE
                )

            pc_range = self.voxel_generator.point_cloud_range
            mask = box_utils.mask_boxes_outside_range(gt_boxes, pc_range)
            gt_boxes = gt_boxes[mask]
            gt_classes = gt_classes[mask]
            gt_names = gt_names[mask]

            # limit rad to [-pi, pi]
            gt_boxes[:, 6] = common_utils.limit_period(gt_boxes[:, 6], offset=0.5, period=2 * np.pi)

        points = points[:, :cfg.DATA_CONFIG.NUM_POINT_FEATURES['use']]
        if cfg.DATA_CONFIG[self.mode].SHUFFLE_POINTS:
            np.random.shuffle(points)

        voxel_grid = self.voxel_generator.generate(points)

        # Support spconv 1.0 and 1.1
        try:
            voxels, coordinates, num_points = voxel_grid
        except:
            voxels = voxel_grid["voxels"]
            coordinates = voxel_grid["coordinates"]
            num_points = voxel_grid["num_points_per_voxel"]

        voxel_centers = (coordinates[:, ::-1] + 0.5) * self.voxel_generator.voxel_size \
                        + self.voxel_generator.point_cloud_range[0:3]

        if cfg.DATA_CONFIG.MASK_POINTS_BY_RANGE:
            points = common_utils.mask_points_by_range(points, cfg.DATA_CONFIG.POINT_CLOUD_RANGE)

        example = {}
        if has_label:
            if not self.training:
                # for eval_utils
                selected = common_utils.keep_arrays_by_name(gt_names, self.class_names)
                gt_boxes = gt_boxes[selected]
                gt_names = gt_names[selected]
                gt_classes = np.array([self.class_names.index(n) + 1 for n in gt_names], dtype=np.int32)


            gt_boxes = np.concatenate((gt_boxes, gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)

            example.update({
                'gt_boxes': gt_boxes
            })

        example.update({
            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coordinates,
            'voxel_centers': voxel_centers,
            'calib': input_dict['calib'],
            'points': points
        })

        return example



    def __len__(self):
        return len(self.kitti_infos)
        # return len(self.sample_idx_list)

    def __getitem__(self, index):

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        points = self.get_lidar(sample_idx)
        calib = self.get_calib(sample_idx)

        img_shape = info['image']['image_shape']
        if cfg.DATA_CONFIG.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        input_dict = {
            'points': points,
            'sample_idx': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            bbox = annos['bbox']
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            if 'gt_boxes_lidar' in annos:
                gt_boxes_lidar = annos['gt_boxes_lidar']
            else:
                gt_boxes_lidar = box_utils.boxes3d_camera_to_lidar(gt_boxes, calib)

            input_dict.update({
                'gt_boxes': gt_boxes,
                'gt_names': gt_names,
                'gt_box2d': bbox,
                'gt_boxes_lidar': gt_boxes_lidar
            })

        example = self.prepare_data(input_dict=input_dict, has_label='annos' in info)

        example['sample_idx'] = sample_idx
        example['image_shape'] = img_shape

        return example

    @staticmethod
    def collate_batch(batch_list,_unused=False):
        example_merged =defaultdict(list)
        for example in batch_list:
            for k,v in example.items():
                example_merged[k].append(v)
        ret ={}

        for key,elems in example_merged.items():
            if key in ["voxels","num_points",
                       "voxel_centers","seg_labels",
                       "part_labels","bbox_reg_labels"]:
                ret[key] = np.concatenate(elems,axis=0)
            elif key in ["coordinates","points"]:
                coors=[]
                for i,coor in enumerate(elems):
                    #第一个（0,0）给coordinates(160000,3)的第一维前面添加0行,后边添加0行
                    #第二个（1,0）代表给列前面添加1列，后边添加0列
                    coor_pad = np.pad(coor,( (0,0), (1,0) ),mode="constant",constant_values=i )
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors,axis=0)
            elif key in ["gt_boxes"]:
                max_gt = 0
                batch_size = elems.__len__()
                for k in range(batch_size):
                    max_gt = max(max_gt, elems[k].__len__())
                batch_gt_boxes3d = np.zeros((batch_size,max_gt,elems[0].shape[-1]))
                for k in range(batch_size):
                    batch_gt_boxes3d[k,:elems[k].__len__(),:] =elems[k]
                ret[key] = batch_gt_boxes3d
            else:
                ret[key] = np.stack(elems,axis=0)
        ret["batch_size"] = batch_list.__len__()

        return ret


    @staticmethod
    def generate_prediction_dict(input_dict,index,record_dict):
        """
        相机坐标系的标注box的dim 是l,h,w
        :param input_dict:
        :param index:
        :param record_dict:
        :return:
        """
        sample_idx = input_dict["sample_idx"][index] if "sample_idx" in input_dict else -1
        boxes3d_lidar_preds = record_dict["boxes"].cpu().numpy()
        if boxes3d_lidar_preds.shape[0] == 0:
            return {"sample_idx": sample_idx}
        calib = input_dict["calib"][index]
        image_shape =input_dict["image_shape"][index]

        boxes3d_camera_preds  =box_utils.boxes3d_lidar_to_camera(boxes3d_lidar_preds,calib)
        boxes2d_image_preds = box_utils.boxes3d_camera_to_imageboxes(boxes3d_camera_preds,calib,image_shape=image_shape)

        #predict dict
        predictions_dict = {
            "bbox":boxes2d_image_preds,
            "box3d_camera":boxes3d_camera_preds,
            "box3d_lidar":boxes3d_lidar_preds,
            "scores":record_dict["scores"].cpu().numpy(),
            "label_preds":record_dict["labels"].cpu().numpy(),
            "sample_idx":sample_idx
        }
        return predictions_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]
            pred_boxes_camera = box_utils.boxes3d_lidar_to_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['sample_idx'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = os.path.join(output_path, ('%s.txt' % frame_id))
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self,det_annos,class_names,**kwargs):
        assert "annos" in self.kitti_infos[0].keys()
        import pvdet.dataset.Dataset.kitti_object_eval.eval as kitti_eval
        #import debug_code.kitti_object_eval_python.eval as kitti_eval
        if "annos" not in self.kitti_infos[0]:
            return "None",{}
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info["annos"]) for info in self.kitti_infos]
        ap_result_str,ap_dict = kitti_eval.get_offical_eval_result(eval_gt_annos,eval_det_annos,class_names)#我写的
        #ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)#源码



        return ap_result_str,ap_dict


if __name__ == '__main__':
    pass
    # path_test_dbinfos ="/home/ubuntu-502/liang/PVRCNN-V1.1/data/kitti/database_information/kitti_dbinfos_train.pkl"
    # with open(path_test_dbinfos,"rb") as f:
    #     info = pickle.load(f)
    # print("done")
    # path = '/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/kitti/training/velodyne/006920.bin'
    # points = np.fromfile(path, dtype=np.float).reshape(-1, 4)
    # print("done")
    # from new_train.config import cfg
    # import torch
    # from torch.utils.data import DataLoader
    # data_path =cfg.DATA_DIR
    # logger = None
    # dataset = KittiDataset(datapath=data_path,
    #                        class_name=cfg.CLASS_NAMES,
    #                        training=True,
    #                        split="train",
    #                        # split = cfg.MODEL["TRAIN" if training else "TEST"].SPLIT,
    #                        logger=logger
    #                        )
    # dist = False
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset) if dist else None
    # dataloader = DataLoader(dataset=dataset,
    #                         batch_size=2,
    #                         pin_memory=True,
    #                         num_workers=15,
    #                         shuffle=(sampler is None) and True,
    #                         collate_fn=dataset.collate_batch,
    #                         drop_last=False,
    #                         sampler=sampler,
    #                         timeout=0
    #                         )
    # data_iter = iter(dataloader)
    # batch_data = next(data_iter)
    # print("done")




