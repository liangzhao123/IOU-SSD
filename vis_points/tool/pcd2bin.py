
# -*- coding: utf-8 -*-
import numpy as np
import glob
import os
import open3d as o3d
import mayavi.mlab as mlab
import vis_points.utils as utils

def load_pcd_data_origial(file_path):
    pts = []
    f = open(file_path, 'r')
    data = f.readlines()

    f.close()
    line = data[9]
    # print line
    line = line.strip('\n')
    i = line.split(' ')
    pts_num = eval(i[-1])
    for line in data[11:]:
        line = line.strip('\n')
        xyzargb = line.split(' ')
        x, y, z = [eval(i) for i in xyzargb[:3]]
        argb = xyzargb[-1]
        # print type(bgra)
        argb = bin(eval(argb))[2:]
        a, r, g, b = [int(argb[8 * i:8 * i + 8], 2) for i in range(4)]
        pts.append([x, y, z, a, r, g, b])

    assert len(pts) == pts_num
    res = np.zeros((pts_num, len(pts[0])), dtype=np.float)
    for i in range(pts_num):
        res[i] = pts[i]
    # x = np.zeros([np.array(t) for t in pts])
    return res

def render_points(points):
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
    )
    fig = utils.draw_lidar(points, fig=fig,color_by_intensity=True)
    mlab.show()

def load_pcd_data(file_path):
    pts = []
    pcd = o3d.io.read_point_cloud(file_path,format="pcd")
    print(np.asarray(pcd.points))
    colors = np.asarray(pcd.colors) * 255
    intensity = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    print(points.shape, colors.shape)
    return np.concatenate([points, colors], axis=-1)
if __name__ == '__main__':
    path = "/media/liang/Elements/rosbag/leishen_e70_32/label-test/pcd"
    pcd_list = glob.glob(os.path.join(path,"*.pcd"))
    pcd_list.sort()
    pcd_filename = pcd_list[0]
    # load_pcd_data_origial(pcd_filename)
    # load_pcd_data(pcd_filename)
    bin_path =  "/media/liang/Elements/rosbag/leishen_e70_32/label-test/pcd/5573.bin"
    points = np.fromfile(bin_path,dtype=np.float32).reshape(-1,4)
    render_points(points)
    print("done")
