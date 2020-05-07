try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'mesh_gen_utils.libkdtree.pykdtree.kdtree',
    sources=[
        'mesh_gen_utils/libkdtree/pykdtree/kdtree.c',
        'mesh_gen_utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
)

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'mesh_gen_utils.libmcubes.mcubes',
    sources=[
        'mesh_gen_utils/libmcubes/mcubes.pyx',
        'mesh_gen_utils/libmcubes/pywrapper.cpp',
        'mesh_gen_utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'mesh_gen_utils.libmesh.triangle_hash',
    sources=[
        'mesh_gen_utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'mesh_gen_utils.libmise.mise',
    sources=[
        'mesh_gen_utils/libmise/mise.pyx'
    ],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'mesh_gen_utils.libsimplify.simplify_mesh',
    sources=[
        'mesh_gen_utils/libsimplify/simplify_mesh.pyx'
    ]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'mesh_gen_utils.libvoxelize.voxelize',
    sources=[
        'mesh_gen_utils/libvoxelize/voxelize.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

# Gather all extension modules
ext_modules = [
    pykdtree,
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    },
    include_dirs=[numpy.get_include()]
)
