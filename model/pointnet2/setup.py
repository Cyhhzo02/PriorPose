# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os

_ext_src_root = "_ext_src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

headers = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), '_ext_src', 'include')

setup(
    name='pointnet2',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", headers],
                "nvcc": ["-O2", headers]
            },
        ),
        CUDAExtension(
            name='pointnet2._ordered',
            sources=['_ordered_src/binding.cpp', '_ordered_src/ordered.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--fmad=false', '--ftz=false'],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
