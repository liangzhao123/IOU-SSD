
import numpy as np

import mayavi.mlab as mlab

from single_stage_model.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
import torch
import cv2
import os
import vis_points.kitti_utils as kitti_utils

def draw_lidar(
    pc,
    color=None,
    fig=None,
    bgcolor=(0, 0, 0),
    pts_scale=0.3,
    pts_mode="sphere",
    pts_color=None,
    color_by_intensity=False,
    pc_label=False,
    datatype="kitti",
    draw_axis=True,
    draw_square_region=True,
    draw_fov=True
):
    """ Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    """
    # ind = (pc[:,2]< -1.65)
    # pc = pc[ind]

    pts_mode = "point"
    print("====================", pc.shape)
    if fig is None:
        fig = mlab.figure(
            figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000)
        )
    if color is None:
        color = pc[:, 2]
    if pc_label:
        color = pc[:, 4]
    if color_by_intensity:
        color = pc[:, 2]

    mlab.points3d(
        pc[:, 0],
        pc[:, 1],
        pc[:, 2],
        color,
        color=(1,1,1),
        mode=pts_mode,
        colormap="gnuplot",
        scale_factor=pts_scale,
        figure=fig,
    )

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2)

    # draw axis
    if draw_axis:
        axes = np.array(
            [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
            dtype=np.float64,
        )
        mlab.plot3d(
            [0, axes[0, 0]],
            [0, axes[0, 1]],
            [0, axes[0, 2]],
            color=(1, 0, 0),
            tube_radius=None,
            figure=fig,
        )
        mlab.plot3d(
            [0, axes[1, 0]],
            [0, axes[1, 1]],
            [0, axes[1, 2]],
            color=(0, 1, 0),
            tube_radius=None,
            figure=fig,
        )
        mlab.plot3d(
            [0, axes[2, 0]],
            [0, axes[2, 1]],
            [0, axes[2, 2]],
            color=(0, 0, 1),
            tube_radius=None,
            figure=fig,
        )

    # draw fov (todo: update to real sensor spec.)
    if draw_fov:
        fov = np.array(
            [[20.0, 20.0, 0.0, 0.0], [20.0, -20.0, 0.0, 0.0]], dtype=np.float64  # 45 degree
        )

        mlab.plot3d(
            [0, fov[0, 0]],
            [0, fov[0, 1]],
            [0, fov[0, 2]],
            color=(1, 1, 1),
            tube_radius=None,
            line_width=1,
            figure=fig,
        )
        mlab.plot3d(
            [0, fov[1, 0]],
            [0, fov[1, 1]],
            [0, fov[1, 2]],
            color=(1, 1, 1),
            tube_radius=None,
            line_width=1,
            figure=fig,
        )

    # draw square region
    if draw_square_region:
        TOP_Y_MIN = -40
        TOP_Y_MAX = 40
        TOP_X_MIN = 0
        TOP_X_MAX = 70.4
        #TOP_Z_MIN = -3
        #TOP_Z_MAX = 2

        x1 = TOP_X_MIN
        x2 = TOP_X_MAX
        y1 = TOP_Y_MIN
        y2 = TOP_Y_MAX
        mlab.plot3d(
            [x1, x1],
            [y1, y2],
            [0, 0],
            color=(0.5, 0.5, 0.5),
            tube_radius=0.1,
            line_width=1,
            figure=fig,
        )
        mlab.plot3d(
            [x2, x2],
            [y1, y2],
            [0, 0],
            color=(0.5, 0.5, 0.5),
            tube_radius=0.1,
            line_width=1,
            figure=fig,
        )
        mlab.plot3d(
            [x1, x2],
            [y1, y1],
            [0, 0],
            color=(0.5, 0.5, 0.5),
            tube_radius=0.1,
            line_width=1,
            figure=fig,
        )
        mlab.plot3d(
            [x1, x2],
            [y2, y2],
            [0, 0],
            color=(0.5, 0.5, 0.5),
            tube_radius=0.1,
            line_width=1,
            figure=fig,
        )

    # mlab.orientation_axes()
    if datatype =="kitti":
        mlab.view(
            azimuth=180,
            elevation=70,
            focalpoint=[12.0909996, -1.04700089, -2.03249991],
            distance=62.0,
            figure=fig,
        )
    elif datatype=="leishen":
        mlab.view(
            azimuth=270,
            elevation=30,
            focalpoint=[12.0909996, -1.04700089, -2.03249991],
            # focalpoint=[5.0909996, -0.5, -1.0],
            # focalpoint=[0.5, -5.0909996, -1.0],
            distance=5,
            figure=fig,
        )
    return fig

def draw_gt_boxes3d(
    gt_boxes3d,
    fig,
    color=(1, 1, 1),
    line_width=1,
    draw_text=False,
    text_scale=(1, 1, 1),
    color_list=None,
    label="",
    draw_dir = True,
    gt_boxes_center=None,
):
    """ Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    """
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text:
            mlab.text3d(
                b[4, 0],
                b[4, 1],
                b[4, 2],
                label[n],
                scale=text_scale,
                color=color,
                figure=fig,
            )

        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k, k + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )
    if draw_dir  is not None:
        for n in range(num):
            b = gt_boxes3d[n]
            if color_list is not None:
                color = color_list[n]
            # x_center = center[0]
            # y_center = center[1]
            # z_center = center[2]
            # x_forward = np.sum(b[:2, 0],axis=0)/2 + 0.5
            # y_forward = np.sum(b[:2, 1],axis=0)/2 + 0.5
            # z_forward = z_center
            for k in range(0, 1):
                # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
                mlab.plot3d(
                    [b[0,0], b[5,0]],
                    [b[0,1], b[5,1]],
                    [b[0,2], b[5,2]],
                    color=color,
                    tube_radius=0.05,
                    line_width=0.5,
                    figure=fig,
                )
                mlab.plot3d(
                    [b[1, 0], b[4, 0]],
                    [b[1, 1], b[4, 1]],
                    [b[1, 2], b[4, 2]],
                    color=color,
                    tube_radius=0.05,
                    line_width=0.5,
                    figure=fig,
                )
    # mlab.show(1)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def points_indices(points,gt_boxes):
    indices = points_in_boxes_cpu(torch.from_numpy(points[:,:3]),torch.from_numpy(gt_boxes[:,:7])).numpy()
    points_in_list = []
    for i,indice in enumerate(indices):
        points_in_list.append(points[indice>0])
    points_in_boxes = np.concatenate(points_in_list,axis=0)
    indices = np.sum(indices,axis=0)
    points_out_boxes = points[indices==0]
    return points_in_boxes,points_out_boxes

def show_image_with_boxes_3type(img, objects, name, objects_pred=None,draw_2d=False):
    """ Show image with 2D bounding boxes
    objects is gt
    """
    img1 = np.copy(img)  # for 2d bbox
    type_list = ["Pedestrian", "Car", "Cyclist"]
    # draw Label
    color = (0, 255, 0)
    if draw_2d:
        for obj in objects:
            if obj.type not in type_list:
                continue
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                color,
                3,
            )
    startx = 5
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_lables = [obj.type for obj in objects if obj.type in type_list]
    text_lables.insert(0, "Label:")
    for n in range(len(text_lables)):
        text_pos = (startx, 25 * (n + 1))
        cv2.putText(img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
    # draw 2D Pred
    color = (0, 0, 255)
    # for obj in objects2d:
    #     cv2.rectangle(
    #         img1,
    #         (int(obj.box2d[0]), int(obj.box2d[1])),
    #         (int(obj.box2d[2]), int(obj.box2d[3])),
    #         color,
    #         2,
    #     )
    startx = 85
    font = cv2.FONT_HERSHEY_SIMPLEX

    # text_lables = [type_list[obj.typeid - 1] for obj in objects2d]
    # text_lables.insert(0, "2D Pred:")
    # for n in range(len(text_lables)):
    #     text_pos = (startx, 25 * (n + 1))
    #     cv2.putText(img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
    # draw 3D Pred

    if objects is not None:
        color = (255, 0, 0)
        for obj in objects:
            if obj.type not in type_list:
                continue
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                color,
                1,
            )
        startx = 165
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_lables = [obj.type for obj in objects if obj.type in type_list]
        text_lables.insert(0, "3D Pred:")
        for n in range(len(text_lables)):
            text_pos = (startx, 25 * (n + 1))
            cv2.putText(
                img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA
            )



    if objects_pred is not None:
        color = (255, 0, 0)
        for obj in objects_pred:
            if obj.type not in type_list:
                continue
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                color,
                1,
            )
        startx = 165
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_lables = [obj.type for obj in objects_pred if obj.type in type_list]
        text_lables.insert(0, "3D Pred:")
        for n in range(len(text_lables)):
            text_pos = (startx, 25 * (n + 1))
            cv2.putText(
                img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA
            )

    cv2.imshow("with_bbox", img1)
    cv2.imwrite("imgs/" + str(name) + ".png", img1)


def show_image_with_boxes(img, objects, calib, pred_object=None,show3d=True, depth=None):
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    # img3 = np.copy(img)  # for 3d bbox
    if objects is None:
        return 0
    for obj in objects:
        if obj.type == "DontCare":
            continue
        cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (0, 255, 0),
            2,
        )
        box3d_pts_2d, _ = kitti_utils.compute_box_3d(obj, calib.P)
        img2 = kitti_utils.draw_projected_box3d(img2, box3d_pts_2d,color=(0,0,255))

        # project
        # box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # box3d_pts_32d = utils.box3d_to_rgb_box00(box3d_pts_3d_velo)
        # box3d_pts_32d = calib.project_velo_to_image(box3d_pts_3d_velo)
        # img3 = utils.draw_projected_box3d(img3, box3d_pts_32d)
    # print("img1:", img1.shape)

    for obj in pred_object:
        if obj.type == "DontCare":
            continue
        cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (0, 0, 255),
            2,
        )
        box3d_pts_2d, _ = kitti_utils.compute_box_3d(obj, calib.P)
        img2 = kitti_utils.draw_projected_box3d(img2, box3d_pts_2d,color=(0,255,0))

    cv2.imshow("2dbox", img1)
    # print("img3:",img3.shape)
    # Image.fromarray(img3).show()
    if show3d:
        # print("img2:",img2.shape)
        cv2.imshow("3dbox", img2)
    if depth is not None:
        cv2.imshow("depth", depth)

    return img1, img2

def show_lidar_with_boxes(
    pc_velo,
    objects,
    calib,
    img_fov=False,
    img_width=None,
    img_height=None,
    objects_pred=None,
    depth=None,
    cam_img=None,
        fig =None,
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """




    print(("All point num: ", pc_velo.shape[0]))
    if fig is None:
        fig = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
        )

    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig)
    # pc_velo=pc_velo[:,0:3]

    color = (1, 0, 0)
    if objects_pred is not None:
        color = (0, 1, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = kitti_utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            label = obj.type
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color,draw_text=True,label=[label])
            # Draw heading arrow
            _, ori3d_pts_3d = kitti_utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    if objects is None:
        mlab.show()
        return 0
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        _, box3d_pts_3d = kitti_utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)

        label = obj.type
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color,draw_text=False,label = [label])

        # Draw depth
        # if depth is not None:
        #     # import pdb; pdb.set_trace()
        #     depth_pt3d = depth_region_pt3d(depth, obj)
        #     depth_UVDepth = np.zeros_like(depth_pt3d)
        #     depth_UVDepth[:, 0] = depth_pt3d[:, 1]
        #     depth_UVDepth[:, 1] = depth_pt3d[:, 0]
        #     depth_UVDepth[:, 2] = depth_pt3d[:, 2]
        #     print("depth_pt3d:", depth_UVDepth)
        #     dep_pc_velo = calib.project_image_to_velo(depth_UVDepth)
        #     print("dep_pc_velo:", dep_pc_velo)
        #
        #     draw_lidar(dep_pc_velo, fig=fig, pts_color=(1, 1, 1))

        # Draw heading arrow
        _, ori3d_pts_3d =kitti_utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        mlab.plot3d(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color=color,
            tube_radius=None,
            line_width=1,
            figure=fig,
        )

    mlab.show()

class kitti_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, split="training", pred_dir=None,proposal_dir=None,args=None):
        """root_dir contains training and testing folders"""
        self.root_dir = root_dir
        self.split = split
        print(root_dir, split)
        self.split_dir = os.path.join(root_dir, split)

        if split == "training":
            self.num_samples = 7481
        elif split == "testing":
            self.num_samples = 7518
        else:
            print("Unknown split: %s" % (split))
            exit(-1)

        lidar_dir = "velodyne"
        depth_dir = "depth"
        # pred_dir = "pred"
        if args is not None:
            lidar_dir = args.lidar
            depth_dir = args.depthdir
            pred_dir = args.preddir

        self.image_dir = os.path.join(self.split_dir, "image_2")
        self.label_dir = os.path.join(self.split_dir, "label_2")
        self.calib_dir = os.path.join(self.split_dir, "calib")

        self.depthpc_dir = os.path.join(self.split_dir, "depth_pc")
        self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
        self.depth_dir = os.path.join(self.split_dir, depth_dir)
        # "/home/liang/for_ubuntu502/PVRCNN-V1.1/output/single_stage_model/0.0.1/eval/epoch_80/final_result/data"
        self.pred_dir = pred_dir
        self.proposal_dir = proposal_dir

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.image_dir, "%06d.png" % (idx))
        return kitti_utils.load_image(img_filename)

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        assert idx < self.num_samples
        lidar_filename = os.path.join(self.lidar_dir, "%06d.bin" % (idx))
        print(lidar_filename)
        return kitti_utils.load_velo_scan(lidar_filename, dtype, n_vec)

    def get_calibration(self, idx):
        assert idx < self.num_samples
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % (idx))
        return kitti_utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert idx < self.num_samples and self.split == "training"
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        return kitti_utils.read_label(label_filename)

    def get_pred_objects(self, idx):
        assert idx < self.num_samples
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return kitti_utils.read_label(pred_filename)
        else:
            return None
    def get_proposals(self,idx):
        assert idx < self.num_samples
        pred_filename = os.path.join(self.proposal_dir, "%06d.txt" % (idx))
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return kitti_utils.read_label(pred_filename)
        else:
            return None

    def get_depth(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.depth_dir, "%06d.png" % (idx))
        return kitti_utils.load_depth(img_filename)

    def get_depth_image(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.depth_dir, "%06d.png" % (idx))
        return kitti_utils.load_depth(img_filename)

    def get_depth_pc(self, idx):
        assert idx < self.num_samples
        lidar_filename = os.path.join(self.depthpc_dir, "%06d.bin" % (idx))
        is_exist = os.path.exists(lidar_filename)
        if is_exist:
            return kitti_utils.load_velo_scan(lidar_filename), is_exist
        else:
            return None, is_exist
        # print(lidar_filename, is_exist)
        # return utils.load_velo_scan(lidar_filename), is_exist

    def get_top_down(self, idx):
        pass

    def isexist_pred_objects(self, idx):
        assert idx < self.num_samples and self.split == "training"
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        return os.path.exists(pred_filename)

    def isexist_depth(self, idx):
        assert idx < self.num_samples and self.split == "training"
        depth_filename = os.path.join(self.depth_dir, "%06d.txt" % (idx))
        return os.path.exists(depth_filename)


if __name__ == '__main__':
    """path = "/home/liang/kitti/training/velodyne/002287.bin"
    import mayavi.mlab as mlab
    points = np.fromfile(path,dtype=np.float32).reshape(-1,4)
    fig = mlab.figure(
        figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000)
    )

    fig = draw_lidar(points,color=points[:,3],fig=fig)
    mlab.show()"""



    """viscls = VisPoints(path)
    pcl_points = viscls.read_points()
    viscls.visial_pcl()"""
