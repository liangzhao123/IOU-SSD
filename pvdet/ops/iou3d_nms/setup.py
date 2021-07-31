from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='iou3d_nms_cuda',
    ext_modules=[
        CUDAExtension('iou3d_nms_cuda', [
            'src/iou3d_cpu.cpp',
            'src/iou3d_nms_api.cpp',
            'src/iou3d_nms.cpp',
            'src/iou3d_nms_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension})
