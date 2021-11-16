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
"""utils.py

This file contains classes and functions for injecting and erasing perfpoint
banners from C/C++ code.
"""

import logging
from copy import deepcopy
from collections import OrderedDict

import numpy as np

from autoperf.annotation.parsing import ParseResult

log = logging.getLogger('rich')

# ------------------------------------------------------------------------------

banner = "// ==== [ INJECTED BY AUTOPERF ] ==== //"

perf_include = \
    """{banner}
#ifndef NO_AUTOPERF
#include "perfpoint.h"
#endif
{banner}
"""

perf_start = \
    """{banner}
#ifndef NO_AUTOPERF
perfpoint_START({mark_id});
#endif
{banner}
"""

perf_end = \
    """{banner}
#ifndef NO_AUTOPERF
perfpoint_END();
#endif
{banner}
"""

annotating = \
    "[bright_blue][code]{file}[/code]: annotating function `[code]{func}[/code]`.[/bright_blue]"

already_annotated = \
    "[bright_blue][code]{file}[/code]: function `[code]{func}[/code]` is already annotated.[/bright_blue]"


def get_perf_include() -> list:
  """Get the `#include "perfpoint.h"` banner."""
  return perf_include.format(banner=banner).splitlines(True)


def get_perf_start(mark_id: int) -> list:
  """Get the `perfpoint_START()` banner.

  Args:
      mark_id: Integer corresponding to a particular function / method.
  """
  return perf_start.format(banner=banner, mark_id=mark_id).splitlines(True)


def get_perf_end() -> list:
  """Get the `perfpoint_END()` banner."""
  return perf_end.format(banner=banner).splitlines(True)


class Injector():
  """Custom class for injecting perfpoint banners into a codebase."""
  mark_id = None
  only = None

  def __init__(self, only: dict = None):
    self.mark_id = 1
    self.only = only

  def inject(self, result: ParseResult):
    """Inject perfpoint `START` and `END` banners into the code.

    Args:
        result: Result of the Clang parser for a particular function / method.
    """
    # Read the file into a list of lines
    with open(result.filename) as f:
      lines = f.readlines()

    # Iterate through the functions / methods / etc. we want to monitor
    new_lines = {0: get_perf_include()}
    print_queue = []

    functions = []
    try:
      functions += list(result.d['functions'].items())
    except KeyError:
      ...
    try:
      functions += list(result.d['methods'].items())
    except KeyError:
      ...

    for _, func in functions:

      # Only annotate function definitions
      if not func['definition']:
        continue

      # Grab the data collected by libclang
      start = func['start']
      end = func['end']

      # Obey the whitelist, if present
      if self.only is not None and len(self.only[str(result.filename)]) > 0:

        # Allow line number indexing rather than by function name
        if func['spelling'] not in self.only[str(result.filename)]:
          present = False
          for line in self.only[str(result.filename)]:
            if start < line < end:
              present = True
              break
          if not present:  # skip the function
            continue

      # Check that the function is not already annotated
      if banner in lines[start + 1]:
        print_queue.append(already_annotated.format(func=func['spelling'], file=result.filename))
        continue

      print_queue.append(annotating.format(func=func['spelling'], file=result.filename))

      # Add a single START() statement at the beginning of the function
      new_lines[start + 1] = get_perf_start(self.mark_id)
      self.mark_id += 1

      # Add an END() statement immediately prior to any return() call
      search = lines[start:end]
      return_found = False
      for i, s in enumerate(search):
        if s.strip().startswith('return'):
          if start + i in new_lines:  # single-line function, don't annotate
            print_queue.pop()
            del new_lines[start + i]
          else:
            new_lines[start + i] = get_perf_end()
          return_found = True

      # If function returns void, there will be no returns - instead, place the END()
      # prior to the final curly brace
      if not return_found:
        new_lines[end - 1] = get_perf_end()

    # Print out the status messages accumulated in the for loop
    for s in print_queue:
      log.info(s)

    # Sort the dictionary by line number
    new_lines = OrderedDict(sorted(new_lines.items()))

    # Iterate backwards so insertion doesn't modify the `line_num`
    # + a slightly hacky way to insert a list into a list: https://stackoverflow.com/a/7376026
    new_file = deepcopy(lines)
    for line_num, text in reversed(new_lines.items()):
      new_file[line_num:line_num] = text

    # Rewrite the file with the injected code
    with open(result.filename, 'w') as f:
      f.write(''.join(new_file))

# ------------------------------------------------------------------------------


class Eraser():
  """Custom class for erasing perfpoint banners from a codebase."""
  # TODO: really don't have to parse into ASTs to erase,
  #       could probably get a big speedup by skipping that

  only = None  # TODO: use this?

  def __init__(self, only):
    self.only = only

  def erase(self, result):
    """Remove perfpoint `START` and `END` banners from a codebase.

    Args:
        result: Result of the Clang parser for a particular function / method.
    """
    with open(result.filename, 'r') as f:
      lines = np.array(f.readlines())

    # Create a boolean mask of injected vs. non-injected content
    flag = False
    mask = np.ones(len(lines), dtype=bool)
    for i, l in enumerate(lines):

      if banner in l:  # Check for presence of our injection banner
        mask[i] = 0
        flag = not flag  # Mask until we hit the second banner

      elif flag:
        mask[i] = 0

    # Apply the mask to the lines to find non-injected regions
    new_lines = lines[mask]

    # Rewrite the file without the injected code
    with open(result.filename, 'w') as f:
      f.write(''.join(new_lines))
