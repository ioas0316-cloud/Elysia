import os
import glob
from setuptools import setup, find_packages, Extension
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

class get_pybind_include:
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

# C++/CUDA source and header paths
INCLUDE_DIRS = [
    get_pybind_include(),
    get_pybind_include(user=True),
    os.path.abspath("include"),
    os.path.abspath("include/elysia/core"),
    os.path.abspath("modules/causal_topology"),
]

SOURCES_CPP = glob.glob("src/core/*.cpp")
SOURCES_CUDA = glob.glob("src/core/*.cu")

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available() and len(SOURCES_CUDA) > 0

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-DWITH_CUDA",
    ],
}

ext_modules = [
    Extension(
        'causal_engine',
        sources=['src/bindings/python_bindings.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            'include',
            'modules/causal_topology',
        ],
        language='c++',
        extra_compile_args=['-O3', '-std=c++17', '-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

if CUDA_AVAILABLE:
    print("[elysia_engine] Compiling CMPLR CUDA Extension...")
    extra_compile_args["cxx"].append("-DWITH_CUDA")
    ext_modules.append(
        CUDAExtension(
            name="elysia_engine._C.cmplr",
            sources=SOURCES_CPP + SOURCES_CUDA,
            include_dirs=INCLUDE_DIRS,
            extra_compile_args=extra_compile_args,
        )
    )
else:
    print("[elysia_engine] CUDA not available or no CUDA sources found. Compiling C++ extension...")
    ext_modules.append(
        CppExtension(
            name="elysia_engine._C.cmplr",
            sources=SOURCES_CPP,
            include_dirs=INCLUDE_DIRS,
            extra_compile_args={"cxx": extra_compile_args["cxx"]},
        )
    )

setup(
    name="elysia_engine",
    version="0.1.0",
    author="Kangdeok Lee",
    description="Non-linear cognitive AI engine core with Clifford Multivector Phase-Lock Relaxation",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
    ],
)
