from setuptools import setup, Extension
import sys
import os

class get_pybind_include:
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

extra_compile_args = ['-O3', '-std=c++17']
extra_link_args = []

sources = [
    'src/bindings/pybind_cl31.cpp'
]

ext_modules = [
    Extension(
        'elysia_cl31_pybind',
        sources=sources,
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            'include',
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='elysia_cl31_pybind',
    version='0.1.0',
    description='Elysia Cl(3,1) STA Engine PyBind Extension',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.6.0'],
    zip_safe=False,
)
