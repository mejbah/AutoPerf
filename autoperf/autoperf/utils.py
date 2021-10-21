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

This file contains some utility function for data access, analyze, and
visualization.
"""

from __future__ import division

import os
from typing import Tuple
import contextlib

import numpy as np
from git import Repo


def compareDataset(datasetArrayBase: np.ndarray, datasetArray: np.ndarray) -> int:
  """Compares the mean values between two datasets.

  Args:
      datasetArrayBase: Original data.
      datasetArray: New data.

  Returns:
      L2 norm of the differences between the input array means.

  """
  meanBase = np.mean(datasetArrayBase, axis=0)
  meanComp = np.mean(datasetArray, axis=0)

  diffVal = np.linalg.norm(meanBase - meanComp)
  return diffVal


def getPerfCounterData(counterId: str, dataDir: str, scale: float) -> Tuple[list, str]:
  """Get the performance counter from the 3rd column of the CSV data file.

  Args:
      counterId: The performance counter name / ID.
      dataDir: The parent folder of the performance counter data.
      scale: Scale factor post-normalization. Accessed via `cfg.training.scale_factor`.

  Returns:
      A list of normalized performance counter results.
      The counter's header.
  """
  filename = dataDir + "/event_" + counterId + "_perf_data.csv"
  perfCounter = []
  with open(filename, 'r') as fp:
    for linenumber, line in enumerate(fp):
      if linenumber == 2:
        headers = line.strip().split(",")  # last one is the counter, 1 and 2  is thd id and instcouunt , 0 is mark id
        datasetHeader = headers[-1]
      if linenumber > 2:
        perfCounters = line.strip().split(",")
        # mark = int(perfCounters[0])  # unused
        # threadCount = int(perfCounters[1])  # unused
        instructionCount = int(perfCounters[2])
        currCounter = int(perfCounters[3])
        normalizedCounter = (currCounter / instructionCount) * scale
        perfCounter.append(normalizedCounter)

  return perfCounter, datasetHeader


def getAutoperfDir(directory: str = None) -> str:
  """Retrieves the full path to the repository's .autoperf directory.

  Args:
      directory: Path within the .autoperf dir.

  Returns:
      str: Path to .autoperf
  """
  repo = Repo(os.getcwd(), search_parent_directories=True)
  if directory:
    return os.path.join(repo.working_tree_dir, '.autoperf', directory)
  return os.path.join(repo.working_tree_dir, '.autoperf')


@contextlib.contextmanager
def set_working_directory(directory: str) -> str:
  """Temporarily cd into the working directory. After context is closed (e.g. the
  function is exited), returns to the previous working directory.

  Args:
      directory: Path to change to.

  Yields:
      New working directory.
  """
  owd = os.getcwd()
  try:
    os.chdir(directory)
    yield directory
  finally:
    os.chdir(owd)
