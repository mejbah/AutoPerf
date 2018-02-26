
from __future__ import division
import matplotlib as mpl
mpl.use('Agg') #instead of Xserver for png
import matplotlib.pyplot as plt
import numpy as np
import os
import errno
import configs
import sys

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


def plotHistogram( datasetList, filename, label="default", secondDatasetList=None, thirdDatasetList=None  ):

  fig = plt.figure()
  ax = fig.add_subplot(111)
  
  #bins = np.linspace(-10, 10, 100)
  bins = 100
  ax.hist(datasetList, bins, alpha=0.5, label='not anomalous')
  if secondDatasetList != None:
    ax.hist(secondDatasetList, bins, alpha=0.5, label='anomalous')
  ax.legend(loc='upper right')
  ax.set_ylabel("sample count")
  ax.set_xlabel("counter value in sample")
  fig.suptitle(label)
  fig.savefig(filename)

def plotDataList( datasetList, filename, label="default", secondDatasetList=None, thirdDatasetList=None  ):
  
  fig = plt.figure()
  ax = fig.add_subplot(111)

  x_values = [x for x in range(len(datasetList))]
  y_values = datasetList 

  
  ax.plot(x_values,y_values, 'bo', label = 'first')
  
  if secondDatasetList != None:

	  x_values = [x for x in range(len(secondDatasetList))]
	  y_values = secondDatasetList

  
	  ax.plot(x_values, y_values, 'ro', label = 'second')

  if thirdDatasetList != None:

	  x_values = [x for x in range(len(thirdDatasetList))]
	  y_values = thirdDatasetList

  
	  ax.plot(x_values, y_values, 'g^', label = 'third')

 
  ax.legend(loc='upper left') 
  ax.set_ylabel(label)
  ax.set_xlabel("executions")

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
"""
def compareDataset( datasetArrayBase, datasetArray ):
  #diff = (datasetArray1Base - datasetArray) ** 2
  #meanDiff = np.mean(diff, axis=0)
  
  diff = (datasetArrayBase - datasetArray)
  # matrix norm
  diffVal = np.linalg.norm(diff)
  return diffVal
"""
"""
take mean of each columns and compare
"""
def compareDataset( datasetArrayBase, datasetArray ):
  meanBase = np.mean(datasetArrayBase, axis=0)
  meanComp = np.mean(datasetArray, axis=0)

  diffVal = np.linalg.norm(meanBase - meanComp)
  return diffVal

"""
get perf counter in 3rd column from the data csv file, 
"""
def getPerfCounterData( counterId, dataDir ):
  filename = dataDir + "/event_" + counterId + "_perf_data.csv"
  perfCounter = []
  with open(filename, 'r') as fp:
    for linenumber,line in enumerate(fp):
	  if linenumber == 2:
		headers = line.strip().split(",")   #last one is the counter, 1 and 2  is thd id and instcouunt , 0 is mark id
		datasetHeader = headers[-1]
	  if linenumber > 2:
		perfCounters = line.strip().split(",")
		mark = int(perfCounters[0])
		threadCount = int(perfCounters[1])
		instructionCount = int(perfCounters[2])
		currCounter = int(perfCounters[3]) 
		#normalizedCounter = ( currCounter / ( instructionCount * threadCount ) )* configs.SCALE_UP_FACTOR
		normalizedCounter = ( currCounter / ( instructionCount ) )* configs.SCALE_UP_FACTOR
		perfCounter.append(normalizedCounter)

  return perfCounter, datasetHeader
	   
def compareCounters( dataDir1, dataDir2, counterId, outputDir=None ): 

  datasetList, datasetHeader1 = getPerfCounterData(counterId, dataDir1)
  secondDatasetList, datasetHeader2 = getPerfCounterData(counterId, dataDir2)
  
  assert datasetHeader1 == datasetHeader2
  plotFile = dataDir1.split('/')[-1] + "_" + datasetHeader1.lstrip()
  if outputDir != None:
    plotFile = outputDir + '/' + plotFile
  print plotFile
  #plotDataList( datasetList, plotFile, datasetHeader1, secondDatasetList )
  plotHistogram( datasetList, plotFile, datasetHeader1, secondDatasetList )
  
  return datasetHeader1


if __name__ == "__main__":

  #plot to compare counters
  first = sys.argv[1]
  second = sys.argv[2]
  counter_id = sys.argv[3]
  output_dir = sys.argv[4] 
  compareCounters(first, second, counter_id, output_dir)
