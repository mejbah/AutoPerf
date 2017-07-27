from __future__ import division
import sys
import climate
import numpy as np
import matplotlib as mpl
mpl.use('Agg') #instead of Xserver for png
import matplotlib.pyplot as plt
import pprint
import numpy as np
import os
import errno
import configs
from utils import *
from random import shuffle
import perfpoint_trainer as pt
from sklearn.cluster import KMeans


topDir = sys.argv[1]
NCLUSTER = 2
if len(sys.argv) > 2:
  NCLUSTER = int(sys.argv[2])

apps = os.listdir(topDir)

meanDatasetList = []
for app in apps:
  print app
  appDir = topDir + "/" + app
  dataDir = appDir + "/no_false_sharing"
  runs = os.listdir(dataDir)
  dataset = None
  for run in runs:
    rundir = dataDir + "/" + run
    datasetHeader, newData = pt.getPerfDataset( rundir , configs.NUMBER_OF_COUNTERS )
    if dataset == None:
      dataset = newData
    else :
      dataset.extend(newData)

  meanDataset = np.mean(dataset, axis=0)
  print meanDataset
  meanDatasetList.append(meanDataset)

clusters = KMeans(n_clusters=NCLUSTER, random_state=0).fit(meanDatasetList)

print clusters.labels_
print clusters.predict(meanDatasetList)
