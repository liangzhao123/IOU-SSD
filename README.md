# Seg-RCNN（LZnet）[blog](https://blog.csdn.net/liang_shuaige/article/details/114854061?spm=1001.2014.3001.5501)
## Segmentation based two stage detector 
### ROS implementation for Seg-RCNN see:
https://v.youku.com/v_show/id_XNDk1MjQxMzg4NA==.html?spm=a2h0c.8166622.PhoneSokuUgc_1.dtitle


# LZnet: two-stage 3D object detector from point cloud
Currently testing in KITTI BEV and 3rd in KITTI 3D.


**Authors**: [liangzhao](https://github.com/liangzhao123)


## Updates
2020-10-10: create this readme file

## Demo
[]()

# Introduction
![]()
A 3D object detector tool , which includes the implementation of Seg-RCNN and IOU-SSD. 
All algrithms are build on the deep learming frame [pytorch](https://pytorch.org/), 

# Dependencies
- `python3.5+`
- `cuda` (version 10.2)
- `torch` (tested on 1.4.0) 
- `torchvision`(tested on 0.5.0)
- `opencv`
- `shapely`
- `mayavi`
- `spconv` (v1.2)

# Installation
### 1. Clone this repository.
```angular2
git clone <XXX.git>
```
### 2. install cuda(10.2) and cudnn(corresponding to cuda)
```
sudo cp include/cudnn.h /usr/local/cuda-10.2/include/
sudo cp lib64/libcudnn* /usr/local/cuda-10.2/lib64/
sudo chmod a+r /usr/local/cuda-10.2/include/cudnn.h
```
### 3. install torch.
```bash
$ pip install torch==1.4.1 torchvision=0.5.0
```
### 4 install spconv.
#### 4.1 download the cmake from offical website the version of cmake should >=3.14
#### 4.2 add a envirment path in bashrc:
```bash
$ export PATH=/home/ubuntu-502/liang/cmake-3.14.0-Linux-x86_64/bin:$PATH
```
#### 4.3 check the cmake version using the following command, if the output is 
```bash
$ cmake --version
```
if the output is following words , the installation of cmake has completed
```
cmake version 3.14.0

CMake suite maintained and supported by Kitware (kitware.com/cmake).
```

4.4 install boost.
```
sudo apt-get install libboost-all-dev
```


4.6 install spconv.(the version cuda) (cuda>10.2,cudnn).
```
cd spconv
python setup.py bdist_wheel
cd dist
pip install spconv-1.2-cp36-cp36m-linux_x86_64.whl
```
4.7 install some commom lib.
```
pip install easydict tensorboardX scikit-image opencv-python tqdm
```
5 install special lib (e.g. roiaware_pool3d_cuda, )
```
cd pvdet/dataset/roiaware_pool3d/
python setup.py install
cd pvdet/ops/iou3d_nms/
python setup.py install
```
install pointnet2
```
cd /pvdet/model/pointnet2/pointnet2_stack
 python set_up.py install
```
install fps_with_features_cuda
```
cd /new_train/ops/fps_wit_forgound_point/
python setup.py install
```

# Data Preparation
1. Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Data to download include:
    * Velodyne point clouds (29 GB): input data to VoxelNet
    * Training labels of object data set (5 MB): input label to VoxelNet
    * Camera calibration matrices of object data set (16 MB): for visualization of predictions
    * Left color images of object data set (12 GB): for visualization of predictions

2. Create cropped point cloud and sample pool for data augmentation, please refer to [SECOND](https://github.com/traveller59/second.pytorch).
```bash
$ python new_train/tools/create_data_info.py
```

3. Split the training set into training and validation set according to the protocol [here](https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz).
```plain
└── DATA_DIR
       ├── training   <-- training data
       |   ├── image_2
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced
       └── testing  <--- testing data
       |   ├── image_2
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced
```

# Pretrained Model
You can download the pretrained model [here](https://drive.google.com/file/d/1WJnJDMOeNKszdZH3P077wKXcoty7XOUb/view?usp=sharing), 
which is trained on the train split (3712 samples) and evaluated on the val split (3769 samples) and test split (7518 samples). 
The performance (using 40 recall poisitions) on validation set is as follows:

|Car  |AP@0.70|, 0.70|, 0.70:|
|----|:----:|:----:|:----:|
|bbox |99.12|96.09|, 93.61|
|bev  |96.55|92.79|, 90.32|
|3d   |91.13|81.54|, 79.71|
# Train
To train the LZnet with single GPU, run the following command:
```
python trainer.py 
```
To train the LZnet with multiple GPUs, run the following command:
```
CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node=4 trainer.py --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 sd_train.py --launcher pytorch
```
# Eval
To evaluate the model, run the following command:
```

```
# observe the training loss
first log in the remote server
```
ssh -L 16006:127.0.0.1:16006 ubuntu-502@10.141.77.234
```
then in the serverce run the tensorboard
```
tensorboard --port=16006 --logdir="/media/ubuntu-502/pan1/liang/PVRCNN-V1.1/output/single_stage_model/train/0.0.2/tensorboard"
tensorboard --port=16006 --logdir="/media/ubuntu-502/pan1/liang/PVRCNN-V1.1/output/single_stage_model/train/0.0.4/tensorboard"
```
finally in local computer open the local web:

# remote file transfer comand:
```
scp -r ubuntu-502@10.141.77.234:/media/ubuntu-502/pan1/liang/PVRCNN-V1.1/ckpt/LZnet/0.0.6/checkpoint_epoch_80.pth /home/liang/for_ubuntu502/PVRCNN-V1.1/ckpt/LZnet/0.0.6/

```

## usage
### 1. 
## Citation
If you find this work useful in your research, please consider cite:
```
@inproceedings{,
title={},
author={},
  booktitle={},
  year={2020}
}
```

## Acknowledgement

* []() 
* []()
* []()
* []()
