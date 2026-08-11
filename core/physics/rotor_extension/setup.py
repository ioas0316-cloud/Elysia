import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Gracefully check if CUDA compiler (nvcc) and GPU are available
cuda_available = torch.cuda.is_available()

# You can also manually disable CUDA compilation if needed via env var
force_cpu = os.getenv("FORCE_CPU", "0") == "1"

ext_modules = []

if cuda_available and not force_cpu:
    print("[*] CUDA detected! Compiling rotor_cuda_ext C++/CUDA extension...")
    ext_modules.append(
        CUDAExtension(
            name='rotor_cuda_ext',
            sources=[
                'csrc/rotor_cuda.cpp',
                'csrc/rotor_cuda_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-Wall'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    )
else:
    print("[!] CUDA not available (or FORCE_CPU=1). Skipping CUDA Extension build and relying on Native PyTorch Fallback.")

setup(
    name='rotor_cuda_ext_pkg',
    version='0.1.0',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    } if ext_modules else {}
)
