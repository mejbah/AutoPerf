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
"""cluster.py

Script to generate clusters of functions.

Example:
    `$ python cluster.py /top/dir num_clusters`
"""

from __future__ import division

import sys
import os
import numpy as np
from sklearn.cluster import KMeans

from autoperf import autoperf
from autoperf.counters import get_num_counters

topDir = sys.argv[1]
NCLUSTER = 2
if len(sys.argv) > 2:
  NCLUSTER = int(sys.argv[2])

apps = os.listdir(topDir)

meanDatasetList = []
for app in apps:
  print(app)
  appDir = os.path.join(topDir, app)
  dataDir = os.path.join(appDir, "no_false_sharing")
  runs = os.listdir(dataDir)
  dataset = None
  for run in runs:
    rundir = os.path.join(dataDir, run)
    datasetHeader, newData = autoperf.getPerfDataset(rundir, get_num_counters())
    if dataset is None:
      dataset = newData
    else:
      dataset.extend(newData)

  meanDataset = np.mean(dataset, axis=0)
  print(meanDataset)
  meanDatasetList.append(meanDataset)

clusters = KMeans(n_clusters=NCLUSTER, random_state=0).fit(meanDatasetList)

print(clusters.labels_)
print(clusters.predict(meanDatasetList))
