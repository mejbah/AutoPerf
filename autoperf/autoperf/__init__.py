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
"""Appends the `libperfpoint.so` dir to $(LIBRARY_PATH) + $(LD_LIBRARY_PATH)."""
import os


def add_environment_variable(key: str, sep: str, val: str):
  """Creates or appends a string to an environment variable.

  Args:
      key: The environment variable name.
      sep: The separator to use if appending a value.
      val: The value to assign to the environment variable.
  """
  if key in os.environ:
    os.environ[key] += sep + val
  else:
    os.environ[key] = val


profiler_path = os.path.join('/'.join(__file__.split('/')[:-2]), 'profiler')

add_environment_variable('LIBRARY_PATH', os.pathsep, profiler_path)
add_environment_variable('LD_LIBRARY_PATH', os.pathsep, profiler_path)
add_environment_variable('CPATH', os.pathsep, profiler_path)
add_environment_variable('C_INCLUDE_PATH', os.pathsep, profiler_path)
add_environment_variable('CPLUS_INCLUDE_PATH', os.pathsep, profiler_path)

add_environment_variable('LIBS', ' ', '-lperfpoint -lpapi -ldl')
add_environment_variable('CFLAGS', ' ', '-lperfpoint -lpapi -ldl')
add_environment_variable('CPPFLAGS', ' ', '-lperfpoint -lpapi -ldl')
add_environment_variable('CXXFLAGS', ' ', '-lperfpoint -lpapi -ldl')

add_environment_variable('TF_CPP_MIN_LOG_LEVEL', '', '3')

# print(f'[DEBUG] loaded the perfpoint.so from {os.environ["LD_LIBRARY_PATH"]}')
