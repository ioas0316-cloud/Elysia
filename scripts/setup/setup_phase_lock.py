import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

sources = [
    'src/phase_lock_engine.cpp',
    'src/sensory_phase_core.cpp',
    'src/bindings/pybind_phase_lock.cpp'
]

include_dirs = [os.path.abspath('include')]
extra_compile_args = {'cxx': ['-O3', '-std=c++20']}

if torch.cuda.is_available():
    sources.append('src/phase_lock_kernel.cu')
    sources.append('src/sensory_phase_core_kernel.cu')
    extra_compile_args['cxx'].append('-DWITH_CUDA')
    extra_compile_args['nvcc'] = ['-O3', '-std=c++20']
    ext_modules = [
        CUDAExtension(
            name='elysia_phase_lock_cuda',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args
        )
    ]
else:
    ext_modules = [
        CppExtension(
            name='elysia_phase_lock_cuda',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args
        )
    ]

setup(
    name='elysia_phase_lock_cuda',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
