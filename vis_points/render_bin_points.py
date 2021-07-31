

import numpy as np
import mayavi.mlab as mlab
import vis_points.utils as utils
import os
import glob
import cv2

def read_img(img_file):
    img_file = cv2.imread(img_file)
    return img_file

def read_bin(path):
    points = np.fromfile(path,dtype=np.float32).reshape(-1, 4)
    return points

def render(points,img):
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
    )
    fig = utils.draw_lidar(points, fig=fig)
    cv2.imshow("image",img)
    mlab.show()
def sort_bin_file():
    bin_dir_sorted = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/rosbag/leishen/2020-12-04-17-00-15-bin/sorted"
    os.makedirs(bin_dir_sorted, exist_ok=True)
    bin_list = glob.glob(os.path.join(bin_dir, "*.bin"))
    bin_list.sort()
    for filename, id in zip(bin_list, range(1, len(bin_list) + 1)):
        points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        new_name = os.path.join(bin_dir_sorted, str(id) + ".bin")
        points.tofile(new_name)

if __name__ == '__main__':
    bin_dir = "/media/liang/Elements/rosbag/leishen_e70_32/bin"
    img_dir = "/media/liang/Elements/rosbag/leishen_e70_32/image/sorted"
    bin_list = glob.glob(os.path.join(bin_dir, "*.bin"))
    img_list = glob.glob(os.path.join(img_dir, "*.png"))
    img_list.sort()
    bin_list.sort()
    j = 0
    for img_file,bin_file,i in zip(img_list,bin_list,range(len(img_list))):
        if i<100:
            continue
        if j>=5:
            points = read_bin(bin_file)
            img = read_img(img_file)
            render(points, img)
            j = 0
        else:
            j+=1


    # for i in bin_list:
    #     points = read_bin(i)
    #     render_points(points)
