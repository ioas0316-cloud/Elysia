from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='elysia_phase_cuda',
    ext_modules=[
        CUDAExtension(
            name='elysia_phase_cuda',
            sources=[
                'multi_scale_binding.cpp',
                'multi_scale_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
