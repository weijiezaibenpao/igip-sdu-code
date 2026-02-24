import os

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

nvcc_std = os.popen("nvcc -h | grep -- '--std'")
nvcc_std = nvcc_std.read()

nvcc_flags = ['-O2', '-allow-unsupported-compiler']
if nvcc_std.__contains__('c++20'):
    nvcc_flags.append('-std=c++20')

setup(
    name='pointnet2',  # used by `pip install`
    version='0.0.1',
    description='',
    cmdclass={
        'build_ext': BuildExtension
    },
    ext_modules=[
        CUDAExtension(
            'pointnet2_cuda',
            [
                'src/pointnet2_api.cpp',
                'src/ball_query.cpp',
                'src/ball_query_gpu.cu',
                'src/group_points.cpp',
                'src/group_points_gpu.cu',
                'src/interpolate.cpp',
                'src/interpolate_gpu.cu',
                'src/sampling.cpp',
                'src/sampling_gpu.cu',
            ],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': nvcc_flags})
    ],
    setup_requires=["pybind11"],
    install_requires=["pybind11"],
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
)
