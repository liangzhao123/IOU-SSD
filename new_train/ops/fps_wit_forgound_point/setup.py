from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fps_with_features',
    ext_modules=[
        CUDAExtension('fps_with_features_cuda', [
            'src/sampling.cpp',
            "src/sampling_gpu.cu",
            "src/python_api.cpp"
        ],
                      extra_compile_args={'cxx': ['-g'],
                                          'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})