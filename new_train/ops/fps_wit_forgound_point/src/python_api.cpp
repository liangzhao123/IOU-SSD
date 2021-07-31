//
// Created by liang on 2020/9/1.
//
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include "sampling.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fps_with_features_wrapper", &fps_with_features_wrapper, "fps_with_features_wrapper");
    m.def("furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper, "furthest_point_sampling_wrapper");
    m.def("gather_points_wrapper", &gather_points_wrapper_fast, "gather_points_wrapper_fast");
}

