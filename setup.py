from distutils.core import setup
from distutils.extension import Extension

setup(name='voxelfeatures',
      version='0.1',
      description='Geometric voxel features for 2D surfaces in 3D',
      author='Dmitry Yarotsky',
      author_email='yarotsky@gmail.com',
      py_modules=['voxelfeatures'],
      ext_modules=[Extension('geomFeatures', 
                             ['cpp/geomFeatures.cpp'],
                             libraries=['armadillo'],
                             include_dirs=['.'],
                             extra_compile_args=['-shared', '-fPIC', '-std=c++11', '-Wall'])],
      )
