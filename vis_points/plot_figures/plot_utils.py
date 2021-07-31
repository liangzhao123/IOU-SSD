#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

subplot_fig_size=(20,30)
subplot_xticks_size=20
subplot_yticks_size=20
subplot_xlabel_size=20
subplot_ylabel_size=20
subplot_title_size=20
subplot_legend_size=20

fig_size = (8, 6)
xticks_size = 20
yticks_size = 20
xlabel_size = 20
ylabel_size = 20
title_size = 20
legend_size = 20
DATA_PATH =  "/home/liang/for_ubuntu502/PVRCNN-V1.1/output/figures/interfer_time_list.pth"
save_DIR = "/home/liang/for_ubuntu502/PVRCNN-V1.1/output/figures"
def plot_iterfer_time():
    runtime_file = os.path.join(save_DIR,"runtime.jpg")
    with open(DATA_PATH,"rb") as f:
        time_list = pickle.load(f)
    time_list = (np.array(time_list[30:]) - 0.03)*1000
    indexs = np.arange(0,len(time_list))
    plt.figure(figsize=fig_size)
    plt.scatter(indexs,time_list,marker="^",color="black",s=20)
    plt.xticks(fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    # plt.xlabel("Frame ID", fontsize=xlabel_size)
    plt.xlabel("帧序号", fontsize=xlabel_size,fontproperties="SimHei")
    plt.ylabel("Runtime [ms]", fontsize=ylabel_size)
    # plt.ylabel("运行时间 [ms]", fontsize=ylabel_size,fontproperties="SimHei")
    # plt.savefig(runtime_file, dpi=600, bbox_inches="tight")
    plt.show()
    print("done")

if __name__ == '__main__':
    import matplotlib

    print(matplotlib.matplotlib_fname())
    plot_iterfer_time()