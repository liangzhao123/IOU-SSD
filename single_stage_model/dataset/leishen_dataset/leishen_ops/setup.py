from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='leishen_ops',
    ext_modules=[
        CUDAExtension('leishen_ops_cuda', [
            'src/roiaware_pool3d.cpp',
            'src/roiaware_pool3d_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension})
