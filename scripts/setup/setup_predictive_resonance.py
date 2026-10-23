import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

sources_cu = 'src/predictive_resonance_kernel.cu'
sources_cpp = 'src/predictive_resonance_kernel.cpp'

if torch.cuda.is_available() and os.system("which nvcc > /dev/null 2>&1") == 0:
    ext = CUDAExtension(
        name='predictive_resonance_cpp',
        sources=[sources_cu],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '-rdc=true', '-lcudadevrt']
        }
    )
else:
    ext = CppExtension(
        name='predictive_resonance_cpp',
        sources=[sources_cpp],
        extra_compile_args=['-O3']
    )

setup(
    name='predictive_resonance_cpp',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension}
)
