#!/usr/bin/env python
# coding=utf-8
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import os
files = [i for i in os.listdir('./') if i.endswith('.py') and i.startswith('hfs_pyalpha')]

ext_modules = [
    Extension(i[:-3],[i],language_level = 3)
    for i in files
]
setup(
        name = "hfs_pyalpha_demo",
        cmdclass = {'build_ext': build_ext},
        ext_modules = ext_modules,
)

# from setuptools import Extension, setup
# from Cython.Build import cythonize
# 
# ext_modules = cythonize([
#         Extension("hfs_pyalpha_demo001",["hfs_pyalpha_demo001.py"],
#                   libraries=["hfs_pyalpha"],
#                   extra_link_args=['-Wl,-rpath=/datas/share/Msimrelease_stkbar/lib'],#一定要加，让运行时候链接器知道去哪儿找文件
#                   include_dirs=['c2/include'],
#                   library_dirs=['c2/lib']
#         ),
# ])
# setup(name='hfs_pyalpha', ext_modules=ext_modules,)
