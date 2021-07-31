#for single_stage model
from single_stage_model.configs.single_stage_config import cfg


import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import pickle


# import pvdet.dataset.utils.box_utils as box_utils
from single_stage_model.dataset import box_utils

# import pvdet.dataset.utils.common_utils as common_utils
from single_stage_model.dataset import common_utils

# from pvdet.dataset.data_augmentation.dbsampler import DataBaseSampler
from single_stage_model.dataset.dbsampler import DataBaseSampler

# from pvdet.dataset.data_augmentation import augmentation_utils
from single_stage_model.dataset import augment_utils as augmentation_utils

import spconv
from collections import defaultdict
import copy
from single_stage_model.detect_head.Anchor_utils import AnchorGenertor

from single_stage_model.dataset.kitti_templte import KittiTemplate
from single_stage_model.utils.geomertry import rbbox2d_to_near_bbox,sparse_sum_for_anchors_mask,fused_get_anchors_area


class KittiDataset(KittiTemplate):
    def __init__(self,datapath=cfg.DATA_DIR,class_name=cfg.CLASS_NAMES,training=True,split="train",logger=None,args=None,train_all=False):
        super().__init__(datapath=datapath,
                 class_name=class_name,
                 training=training,split=split,
                 logger=logger,
                    train_all=train_all)

        self.train_all = train_all
        if os.path.exists(cfg.data_config.augmentation.db_sampler.db_info_path[0]):
            self.db_sampler_init(class_name, logger)
        self.voxel_generator_init()
        self.point_cloud_range = cfg.data_config.point_cloud_range
        range = np.array(np.array(self.point_cloud_range[3:],dtype=np.float32) - np.array(self.point_cloud_range[:3],dtype=np.float32))
        self.voxel_size = self.voxel_generator.voxel_size
        self.grid_size =np.round(range/self.voxel_size).astype(np.int32)
        self.num_point_features =cfg.data_config.num_used_features
        self.kitti_infos = []
        self.include_kitti_data(self.mode,logger,args)
        if cfg.data_config.anchor_mask_enable:
            self.init_anchors(cfg.model.detection_head.anchor_generator)
            self.anchor_area_threshold = cfg.data_config.anchor_mask_config["anchor_area_threshold"]
        else:
            self.anchors =None
            self.anchor_area_threshold = -1

    def init_anchors(self, anchor_generator_config):
        feature_map_size = np.array((np.array(self.point_cloud_range[3:], dtype=np.float32) - np.array(
            self.point_cloud_range[:3], dtype=np.float32)) \
                                    / np.array(cfg.data_config.VoxelGenerator.voxel_size, dtype=np.float32),
                                    dtype=np.int)


        anchor_generator = AnchorGenertor(self.point_cloud_range,
                                          anchor_generator_config)

        features_map_size = [feature_map_size[:2] // config["feature_map_stride"] for
                             config in anchor_generator_config]
        anchors_list, num_anchors_per_location_list = anchor_generator.generator(features_map_size)
        self.anchors = anchors_list #[1,200,176,1,2,7]
        anchor_bv_list = []
        for cls_id in range(len(anchors_list)):
            cur_anchors = anchors_list[cls_id].reshape(-1,7)
            cur_anchor_bv = rbbox2d_to_near_bbox(
                cur_anchors[:, [0, 1, 3, 4, 6]])
            anchor_bv_list.append(cur_anchor_bv)
        self.anchors_bv = anchor_bv_list #[3,70400,4]

        self.features_map_size = features_map_size


    def include_kitti_data(self,mode,logger,args):
        if args.local_rank == 0 and logger is not None:
            logger.info("Loading Kitti Dataset")
        kitti_infos =[]
        info_paths = []
        if self.split=="train":
            info_paths = cfg.data_config.TRAIN.info_path
        elif self.split == "val":
            info_paths = cfg.data_config.TEST.info_path
        elif self.split == "test":
            info_paths = cfg.data_config.offical_test_info_path
        elif self.split == "train_80%":
            info_paths = cfg.data_config.TRAIN.INFO_PATH_TRAIN_80
        elif self.split == "val_20%":
            info_paths = cfg.data_config.TRAIN.INFO_PATH_VAL_20
        elif self.train_all:
            info_paths = [cfg.data_config.TRAIN.info_path[1]]
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
        db_sampler_cfg =cfg.data_config.augmentation.db_sampler
        if self.training and db_sampler_cfg.enable:
            db_infos =[]
            db_info_path = db_sampler_cfg.db_info_path[0]
            if self.train_all:
                db_info_path = db_sampler_cfg.db_info_path[1]
            elif self.split  == "train_80%":
                db_info_path = db_sampler_cfg.db_info_path[2]
            with open(db_info_path,"rb") as f:
                infos=pickle.load(f)
                if db_infos.__len__()==0:
                    db_infos=infos
                    print("len(db_infos):",len(db_infos))
                else:
                    [db_infos[cls].extend(infos) for cls in db_infos.keys()]
            self.db_sampler = DataBaseSampler(
                db_infos=db_infos,sampler_cfg=db_sampler_cfg,class_names=class_name,logger=logger)


    def voxel_generator_init(self):
        voxel_generator_cfg = cfg.data_config.VoxelGenerator
        points = np.zeros((1,4))
        self.voxel_generator = spconv.utils.VoxelGeneratorV2(
            voxel_size=voxel_generator_cfg.voxel_size,
            point_cloud_range=cfg.data_config.point_cloud_range,
            max_num_points=voxel_generator_cfg.max_point_per_voxel,
            max_voxels= cfg.data_config[self.mode].max_number_of_voxel
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
                    if cfg.data_config.augmentation.db_sampler.USE_ROAD_PLANE else None
                sampled_dict = self.db_sampler.sample_all(
                    self.root_path, gt_boxes, gt_names, road_planes=road_planes,
                    num_point_features=cfg.data_config.num_used_features, calib=calib
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

            noise_per_object_cfg = cfg.data_config.augmentation.NOISE_PER_OBJECT
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

            noise_global_scene = cfg.data_config.augmentation.NOISE_GLOBAL_SCENE
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

        points = points[:, :cfg.data_config.num_used_features]
        if cfg.data_config[self.mode].SHUFFLE_POINTS:
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

        if cfg.data_config.mask_point_by_range:
            points = common_utils.mask_points_by_range(points, cfg.data_config.point_cloud_range)


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


        if self.anchor_area_threshold >= 0 and self.anchors is not None:
            voxel_size = self.voxel_generator.voxel_size
            grid_size_bv = tuple(self.grid_size[::-1][1:])
            dense_voxel_map = sparse_sum_for_anchors_mask(
                coordinates, grid_size_bv)

            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            anchor_mask_list =[]

            for cls_anchor_id in range(len(self.anchors)):
                cur_anchor_bv = self.anchors_bv[cls_anchor_id]
                anchors_area = fused_get_anchors_area(
                    dense_voxel_map, cur_anchor_bv, voxel_size, np.array(self.point_cloud_range), self.grid_size)
                anchors_mask = anchors_area > self.anchor_area_threshold
                num_valid_anchor = anchors_mask[anchors_mask==True]
                anchor_mask_list.append(anchors_mask)
            example['anchor_masks'] = anchor_mask_list

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
        if cfg.data_config.fov_points_only:
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
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None,proposal_path=None):
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
            proposal_dict = None
            if box_dict.get("proposals_cls",None) is not None:
                proposal_scores = box_dict["proposals_score"].cpu().numpy()
                proposal_boxes = box_dict["box_proposals"].cpu().numpy()
                proposal_labels = box_dict['proposals_cls'].cpu().numpy()

                proposal_dict = get_template_prediction(proposal_scores.shape[0])
                if proposal_scores.shape[0] !=0:
                    calib = batch_dict['calib'][batch_index]
                    image_shape = batch_dict['image_shape'][batch_index]
                    proposal_boxes_camera = box_utils.boxes3d_lidar_to_camera(proposal_boxes, calib)
                    proposal_boxes_img = box_utils.boxes3d_camera_to_imageboxes(
                        proposal_boxes_camera, calib, image_shape=image_shape
                    )

                    proposal_dict['name'] = np.array(class_names)[proposal_labels - 1]
                    proposal_dict['alpha'] = -np.arctan2(-proposal_boxes[:, 1], proposal_boxes[:, 0]) + proposal_boxes_camera[:, 6]
                    proposal_dict['bbox'] = proposal_boxes_img
                    proposal_dict['dimensions'] = proposal_boxes_camera[:, 3:6]
                    proposal_dict['location'] = proposal_boxes_camera[:, 0:3]
                    proposal_dict['rotation_y'] = proposal_boxes_camera[:, 6]
                    proposal_dict['score'] = proposal_scores
                    proposal_dict['boxes_lidar'] = proposal_boxes


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

            return pred_dict,proposal_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['sample_idx'][index]

            single_pred_dict,single_proposal_dict = generate_single_sample_dict(index, box_dict)
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

            if proposal_path is not None:
                cur_proposal_file = os.path.join(proposal_path, ('%s.txt' % frame_id))
                with open(cur_proposal_file, 'w') as f:
                    bbox = single_proposal_dict['bbox']
                    loc = single_proposal_dict['location']
                    dims = single_proposal_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_proposal_dict['name'][idx], single_proposal_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_proposal_dict['rotation_y'][idx],
                                 single_proposal_dict['score'][idx]), file=f)

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
    # path_test_dbinfos ="/home/ubuntu-502/liang/PVRCNN-V1.1/data/kitti/database_information/kitti_dbinfos_train.pkl"
    # with open(path_test_dbinfos,"rb") as f:
    #     info = pickle.load(f)
    # print("done")
    # path = '/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/kitti/training/velodyne/006920.bin'
    # points = np.fromfile(path, dtype=np.float).reshape(-1, 4)
    # print("done")
    from single_stage_model.configs.single_stage_config import cfg
    import torch
    from torch.utils.data import DataLoader
    from easydict import EasyDict as edict
    args = edict()
    args.local_rank = 0

    data_path =cfg.DATA_DIR
    logger = None
    dataset = KittiDataset(datapath=data_path,
                           class_name=cfg.CLASS_NAMES,
                           training=True,
                           split="train",
                           # split = cfg.MODEL["TRAIN" if training else "TEST"].SPLIT,
                           logger=logger,
                           args=args
                           )
    dist = False
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if dist else None
    dataloader = DataLoader(dataset=dataset,
                            batch_size=2,
                            pin_memory=True,
                            num_workers=15,
                            shuffle=(sampler is None) and True,
                            collate_fn=dataset.collate_batch,
                            drop_last=False,
                            sampler=sampler,
                            timeout=0
                            )
    data_iter = iter(dataloader)
    batch_data = next(data_iter)
    print("done")





