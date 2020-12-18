from setuptools import Extension, setup
from Cython.Build import cythonize

sourcefiles = ['pfunc.pyx']

extensions = [Extension("MulOps", sourcefiles)]

setup(
    name='MulOps',
    ext_modules=cythonize(extensions),
    zip_safe=False,
)