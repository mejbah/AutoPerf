# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# ******************************************************************************
# Copyright (c) 2018 Mejbah ul Alam, Justin Gottschlich, Abdullah Muzahid
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ******************************************************************************

import subprocess

from setuptools import setup
from setuptools import find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


class CustomInstall(install):
  """Custom handler for the 'install' command."""
  def run(self):
    print('[DEBUG] making perfpoint.so')
    subprocess.check_call('make', cwd='./autoperf/profiler/', shell=True)
    super().run()


class CustomDevelop(develop):
  """Custom handler for the 'develop' command."""
  def run(self):
    print('[DEBUG] making perfpoint.so')
    subprocess.check_call('make', cwd='./autoperf/profiler/', shell=True)
    super().run()


setup(name='autoperf',
      version='1.0',
      description='AutoPerf helps identify performance regressions in large codebases',
      long_description='AutoPerf is a tool for low-overhead, automated diagnosis \
                        of performance anomalies in multithreaded programs via \
                        hardware performance counters (HWPCs) in Intel CPUs',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.8',
          'Operating System :: POSIX :: Linux',
          'Topic :: Software Development :: Quality Assurance',
      ],
      keywords='autoperf performance regression monitoring',
      author='Intel Corporation',
      license='MIT',
      # packages=['autoperf','annotation'], #include fsm and annotation
      packages=find_packages(where=".", exclude=("./docs",'./profiler', './.empty', './__pycache__')),
      zip_safe=False,
      entry_points={'console_scripts': ['autoperf=autoperf.__main__:main']},
      cmdclass={'install': CustomInstall, 'develop': CustomDevelop},
      package_dir={'autoperf': 'autoperf'},
      package_data={'autoperf': ['profiler/perfpoint.so']},
      include_package_data=True
)
