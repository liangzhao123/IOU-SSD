//
// Created by liang on 2020/8/29.
//

//#ifndef CUDA_PROJECT_SAMPLING_H
//#define CUDA_PROJECT_SAMPLING_H
// installl for python
#ifndef _SAMPLING_H
#define _SAMPLING_H


//#include <math.h>
//#include <stdio.h>
#include <iostream>
//cuda
#include "cuda_runtime.h"
#include  "cuda_runtime_api.h"

#include "device_launch_parameters.h"
#include <pthread.h>
#include <torch/serialize/tensor.h>
//#include <torch/extension.h>
#include "cuda.h"
//libtorch

#include <ATen/ATen.h>
#include "c10/util/ArrayRef.h"
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>//treath used
//math

#include "vector"
#include <cmath>

//#include "NumCpp.hpp" //


//used for cuda


#include "time.h"
#include "math.h"



//void point_group();
//void test_the_cuda_add_fun();
//at::Tensor fps();
//void pub_fps_cloud(int argc, char **argv);
//void pub_raw_cloud(int argc, char **argv);



//void add_cuda(torch::Tensor a,torch::Tensor b,torch::Tensor c);

//void AddLaunch(const float *a,const float *b,float *c, int size[],cudaStream_t stream);

void fps_cpu(int b,int n, int m,  float *points, float *temp, int *index );



inline int opt_n_threads(int work_size);

void furthest_point_sampling_kernel_launcher(int b, int n, int m,
                                             const float *dataset, float *temp, int *idxs, cudaStream_t stream);


void gather_points_kernel_launcher_fast(int b, int c, int n, int npoints,
                                        const float *points, const int *idx, float *out, cudaStream_t stream);

int gather_points_wrapper_fast(int b, int c, int n, int npoints,
                               at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor);

int furthest_point_sampling_wrapper(int b, int n, int m,
                                at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor);

void further_point_sampling_with_features_kernel_launch(int b, int n, int m,int m1, int c,
                                                        const float *dataset,const  int *predix, float *temp, int *idxs, float *values1,
                                                        float *values2, cudaStream_t stream);

int fps_with_features_wrapper(int b, int n, int m,int m1, int c,
                      at::Tensor points_tensor, at::Tensor predix_tensor,at::Tensor temp_tensor, at::Tensor idx_tensor);




#endif //CUDA_PROJECT_SAMPLING_H
