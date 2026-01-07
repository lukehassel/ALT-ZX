
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='mpo_cuda_ext',
    ext_modules=[
        CUDAExtension('mpo_cuda_ext', [
            'mpo_cuda.cpp',
            'mpo_cuda_kernel.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--use_fast_math']
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
