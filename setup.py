#!/usr/bin/env python

# Copyright (c) 2012 SEOmoz
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os.path
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def find_libxml2_include():
    include_dirs = []
    for d in ['/usr/include/libxml2', '/usr/local/include/libxml2']:
        if os.path.exists(os.path.join(d, 'libxml/tree.h')):
            include_dirs.append(d)
    return include_dirs


class CustomBuildExt(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)

        __builtins__.__NUMPY_SETUP__ = False

        import Cython
        import lxml
        import numpy

        include_dirs = [numpy.get_include()] + lxml.get_include() + find_libxml2_include()

        self.include_dirs.extend(include_dirs)

ext_modules = [
    Extension('dragnet.lcs',
              sources=["dragnet/lcs.cpp"],
              language="c++"),
    Extension('dragnet.blocks',
              sources=["dragnet/blocks.cpp"],
              language="c++",
              libraries=['xml2']),
    Extension('dragnet.features._readability',
              sources=["dragnet/features/_readability.cpp"],
              extra_compile_args=['-std=c++11'] + (['-mmacosx-version-min=10.9'] if sys.platform.startswith("darwin") else []),
              language="c++"),
    Extension('dragnet.features._kohlschuetter',
              sources=["dragnet/features/_kohlschuetter.cpp"],
              language="c++"),
    Extension('dragnet.features._weninger',
              sources=["dragnet/features/_weninger.cpp"],
              language="c++"),
]

setup(
    name='dragnet',
    version='2.0.4',
    description='Extract the main article content (and optionally comments) from a web page',
    author='Matt Peters, Dan Lecocq',
    author_email='matt@moz.com, dan@moz.com',
    url='http://github.com/seomoz/dragnet',
    license='MIT',
    platforms='Posix; MacOS X',
    keywords='automatic content extraction, web page dechroming, HTML parsing',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 5 - Production/Stable',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    packages=['dragnet', 'dragnet.features'],
    package_dir={'dragnet': 'dragnet', 'dragnet.features': 'dragnet/features'},
    package_data={'dragnet': ['pickled_models/*/*']},
    cmdclass={'build_ext': CustomBuildExt},
    ext_modules=ext_modules,
    setup_requires=[
        'Cython>=0.21.1',
        'lxml',
        'numpy'
    ],
    install_requires=[
        'Cython>=0.21.1',
        'ftfy>=4.1.0,<5.0.0',
        'lxml',
        'numpy>=1.11.0',
        'scikit-learn>=0.15.2,<0.21.0',
        'scipy>=0.17.0',
    ]
)
