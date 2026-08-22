from setuptools import setup, Extension
import sys
import os

class get_pybind_include:
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

extra_compile_args = ['-O3', '-std=c++17', '-fopenmp']
extra_link_args = ['-fopenmp']

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
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='causal_engine',
    version='0.1.0',
    author='Elysia Causal Intelligence Engine',
    description='C++ High-Performance Bi-directional Causal Engine Python Binding',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.6.0'],
    install_requires=['numpy'],
    zip_safe=False,
)
