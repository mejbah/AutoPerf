import matplotlib as mpl
mpl.use('Agg') #instead of Xserver for png
import matplotlib.pyplot as plt
import numpy as np
import os
import errno


def mkdir_p(path):
  try:
	os.makedirs(path)
  except OSError as exc:  # Python >2.5
	if exc.errno == errno.EEXIST and os.path.isdir(path):
	  pass
	else:
	  raise


def plotDataArray( datasetArray, filename ):
  
  fig = plt.figure()
  ax = fig.add_subplot(111)

  x_values = np.arange(datasetArray.shape[0])
  y_values = datasetArray[:,1] #first column

  
  ax.plot(x_values,y_values)
  fig.savefig(filename)


def plotDataList( datasetList, filename, label="default", secondDatasetList=None, thirdDatasetList=None  ):
  
  fig = plt.figure()
  ax = fig.add_subplot(111)

  x_values = [x for x in range(len(datasetList))]
  y_values = datasetList 

  
  ax.plot(x_values,y_values, 'bo')
  
  if secondDatasetList != None:

	x_values = [x for x in range(len(secondDatasetList))]
	y_values = secondDatasetList

  
	ax.plot(x_values, y_values, 'ro')

  if thirdDatasetList != None:

	x_values = [x for x in range(len(thirdDatasetList))]
	y_values = thirdDatasetList

  
	ax.plot(x_values, y_values, 'g^')

  
  ax.set_ylabel(label)

  fig.savefig(filename)



def plotPerfCounters( datasetArray1, datasetArray2, column, columnName, outputDir ):
  
  fig = plt.figure()
  ax = fig.add_subplot(111)

  x_values = np.arange(datasetArray1.shape[0])
  y_values = datasetArray1[:,column] 
  
  ax.plot(x_values, y_values, 'bo')

  x_values = np.arange(datasetArray2.shape[0])
  y_values = datasetArray2[:,column] 
  
  ax.plot(x_values, y_values, 'ro')
  ax.set_ylabel(columnName)

  fig.savefig( outputDir + '/counter_'+ columnName + '.png')
  plt.close(fig)


"""
returns mean difference between two dataset
assumes size of two datasets are equal
"""
def compareDataset( datasetArrayBase, datasetArray ):
  #diff = (datasetArray1Base - datasetArray) ** 2
  #meanDiff = np.mean(diff, axis=0)
  
  diff = (datasetArrayBase - datasetArray)
  # matrix norm
  diffVal = np.linalg.norm(diff)
  return diffVal
  
  
  
