# 2015-08-13
# First login as root with sudo su
# Add /usr/lib/x86_64-linux-gnu to PYTHONPATH
# To run: python setup_fast.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("clusters", ["clusters.pyx"],include_dirs=[numpy.get_include()])]
)
