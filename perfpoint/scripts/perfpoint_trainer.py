#!/usr/bin/python

from __future__ import division
import sys
import climate
import numpy as np
import matplotlib as mpl
mpl.use('Agg') #instead of Xserver for png
import matplotlib.pyplot as plt

import pprint

import theanets
import numpy as np
import plot_utils
import os
import errno
import configs
from utils import *
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.metrics import mean_squared_error
## log results ###
#climate.enable_default_logging()




"""
returns number of executions profile csv file present in the directory
"""
def getNumberOfExecProfile( dirName ):
  profiles = [ file.endswith(".csv") for file in dirName]
  return len(profiles)

def getExecProfileFileNames( dirName ):
  profiles = [ file.endswith(".csv") for file in dirName]
  return len(profiles)



def getPerfDataset( dirName , numberOfCounters ):
  datasetHeader = []
  dataset = []
	
  for i in range(1, numberOfCounters):
	if i==15 or  i==16  : 
	  continue #TODO: 2 counters are not set in PAPI, temp fix , remove this once problem is resolved
	filename = dirName + "/event_" + str(i) + "_perf_data.csv"
	with open(filename, 'r') as fp:
	  for linenumber,line in enumerate(fp):
		if linenumber == 2:
		  headers = line.strip().split(",")   #last one is the counter, 1 and 2  is thd id and instcouunt , 0 is mark id
		  datasetHeader.append(headers[-1])
		if linenumber > 2:
		  perfCounters = line.strip().split(",")
		  mark = int(perfCounters[0])
		  threadCount = int(perfCounters[1])
		  instructionCount = int(perfCounters[2])
		  currCounter = int(perfCounters[3])
		  
		  normalizedCounter = currCounter / ( instructionCount * threadCount )
		  if i==1:
			newSample = []
			newSample.append(normalizedCounter)
			dataset.append(newSample)
		  else:
			#print ":::DEBUG::: ", len(dataset), i, linenumber
			##TODO: remove the following hack: added for mysql as files in different counternumber has different total sample
			## hack begin: find the minimum sample in an execution and match that with others
			if len(dataset) < linenumber-2:
			  break; # this hack will create dataset element with smaller vecotors we have to ignore them later
			## hack end
			dataset[linenumber-3].append(normalizedCounter)
  ##TODO: remove the following hack: added for mysql as files in different counternumber has different total sample
  ## hack begin: remove the smaller dataset vectors
  count = 0
  for datavector in dataset:
	if len(datavector) ==  len(dataset[0]):
	  count += 1
	else:
	  break
  dataset = dataset[:count]
  ##hack end	
  
  return datasetHeader, dataset
			

def getDatasetArray( dataset ):
  
  dataArray = np.array(dataset, dtype='float32')
  return dataArray

#def getMSE(old, new):
#  return mean_squared_error(old,new)

def getNormalizedDistance( old, new ):
  dist = np.linalg.norm(old-new)
  origin = np.linalg.norm(old)
  return dist/origin



"""
calculate reconstruction error : normalized distance 
"""
def detectAnomalyPoints( realData, predictedData, outFile, datasetHeader, thresholdLoss ):
  #for x  in realData:
  datasetLen = realData.shape[0]
  dataLen = realData.shape[1]
  anomalyCount = 0
  for x  in range(datasetLen):
	reconstructionError = getNormalizedDistance( realData[x], predictedData[x] )
	#reconstructionError = getMSE( realData[x], predictedData[x] )
	#if(reconstructionError > configs.THRESHOLD_ERROR):
	if(reconstructionError > thresholdLoss):
	  anomalyCount += 1
	  outputStr = "[" + str(x) + ":" + str(reconstructionError)  + "] " #sample number starting from 0
	  #outputStr += str(reconstructionError)
	  for y in range(0, dataLen):
		dist=abs(predictedData[x][y]-realData[x][y]) / realData[x][y]
		if dist > thresholdLoss:
		  outputStr += datasetHeader[y]
		  outputStr += ":"
		  outputStr += str(dist)
		  outputStr += " "
	  outFile.write(outputStr)
	  outFile.write("\n")
  return dataLen, anomalyCount

 

##plotDataset( trainingDataset, testDataset, testHeaderset, outputDir ,reconstructedData )
def plotDataset( datasetArray1, datasetArray2,  datasetHeader, outputDir, datasetArray3=None ):
	
  count = len(datasetHeader)

  for column in range(count):
	columnName = datasetHeader[column]  
    
	fig = plt.figure()
	ax = fig.add_subplot(111)

	x_values = np.arange(datasetArray1.shape[0])
	y_values = datasetArray1[:,column] 
	
	ax.plot(x_values, y_values, 'bo')

	x_values = np.arange(datasetArray2.shape[0])
	y_values = datasetArray2[:,column] 
	
	ax.plot(x_values, y_values, 'ko')


	if datasetArray3 is not None:
	  x_values = np.arange(datasetArray3.shape[0])
	  y_values = datasetArray3[:,column] 
	
	  ax.plot(x_values, y_values, 'rs')
	  
	ax.set_ylabel(columnName)
	
	fig.savefig( outputDir + '/counter_'+ columnName + '.png')
	plt.close(fig)



""""
TODO: train -> each execution file
for train, valid in net.itertrain(train_data, valid_data, **kwargs):
    print('training loss:', train['loss'])
    print('most recent validation loss:', valid['loss'])
"""

def trainAutoencoder( networkConfig, trainingDataArray, validationDataArray, logFile=None, model=None ):
  inputVectorLen = trainingDataArray.shape[1]
  outputVectorLen = trainingDataArray.shape[1]

  assert int(networkConfig[0]) == inputVectorLen
  assert int(networkConfig[-1][1]) == outputVectorLen
  
  if model == None:
	model = theanets.Autoencoder(networkConfig) #data array with 14 columns
	## tied weights has to have palindromic network
	#model = theanets.Autoencoder([inputVectorLen, ( 4,'relu'), ('tied',outputVectorLen)]) #data array with 14 columns
	#model = theanets.Autoencoder([inputVectorLen, (8,'relu'), ( 4,'relu'), ('tied', 8,'relu'), ('tied',outputVectorLen)]) #data array with 14 columns
	#model = theanets.Autoencoder([inputVectorLen, (12,'relu'), ( 8,'relu'), ('tied', 12,'relu'), ('tied',outputVectorLen)]) #data array with 14 columns
	#model = theanets.Autoencoder([inputVectorLen, (12,'relu'), ( 6,'relu'), ('tied', 12,'relu'), ('tied',outputVectorLen)]) #data array with 14 columns
	#model = theanets.Autoencoder([inputVectorLen, (12,'relu'),(8,'relu'), ( 4,'relu'), ('tied', 8,'relu'),('tied', 12, 'relu'), ('tied',outputVectorLen)]) #data array with 14 columns
	if logFile != None:
	  print >> logFile, networkConfig 
  # optional: set up additional losses.
  #model.add_loss('mae', weight=0.1)
  #model.set_loss('mse')
  print "Training autoencoder with dateset " , trainingDataArray.shape[0] , "x" , trainingDataArray.shape[1]
  train_loss = 0
  #for train, valid in model.itertrain(trainingDataArray, validationDataArray):
  train, valid = model.train(trainingDataArray, validationDataArray, algo='layerwise')
  if logFile != None :
	  print >>logFile, train['loss'], valid['err']
  print train['loss'], train['err'] 
  train_loss = train['loss']

  # 2. train the model.
  #In one method, zero-mean Gaussian noise is added to the input data or hidden representations. 
  #These are specified during training using the input_noise and hidden_noise keyword arguments, respectively.
  #The value of the argument specifies the standard deviation of the noise.

  #model.train([trainingDataArray], input_noise=0.1, hidden_noise=0.01)
  #model.train([trainingDataArray])

  #model.train(trainingDataArray, 
          #algo='layerwise',
          #train_batches=10, #trainingDataArray.shape[0],
          #learning_rate=0.01,
          #momentum=0.9,
          #hidden_dropout=0.5)
		  #input_noise=0.1)


  #score = model.score(trainingDataArray)
  #model.train([trainingDataArray], hidden_l1=0.1 ) #sparsity penalty
  #model.train(
  #    training_data,
  #    validation_data,
  #    algo='rmsprop',
  #    hidden_l1=0.01,  # apply a regularizer.
  #)
  return model, train_loss



def runTrainedAutoencoder( model, testDataArray, datasetHeader, thresholdLoss, outFile ):
  
  #print(model.score(testDataArray))
  
  
  encoded_data = model.encode(testDataArray)
  decoded_data = model.decode(encoded_data)
  
  dataLen, anomalyCount = detectAnomalyPoints(testDataArray, decoded_data, outFile, datasetHeader, thresholdLoss)
  
  return decoded_data, anomalyCount


def createDataArray( datapath, numberofexec ):

  topLevelDirName = datapath + "/outputs"
  
  print "reading dataset\n"    
  dataset, datasetHeader = createDataset(numberofexec, topLevelDirName)

  print "dataset created\n"    
  #make np array
  dataArray = np.array(dataset, dtype='float32')

  return dataArray , datasetHeader
 
  

def preprocessDataArray( dataset ):
  #zero centering
  #dataset -= np.mean(dataset, axis=0)
  #normalize
  #dataset /= np.std(dataset, axis=0)

  return dataset


def analyzeVariationInData( dataDir, testDir=None, validationDir=None ):
  
  runs = os.listdir(dataDir)
  results = [0]
  baseDataset = None
  for counter, run in enumerate(runs):
  	datadir = dataDir + "/" + run
  	datasetHeader, dataset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )
	if counter == 0:
  	  baseDataset = getDatasetArray(dataset)
	else:
	  if counter < 50:
		results.append(compareDataset(baseDataset, getDatasetArray(dataset)))

  testResults = []
  if testDir != None:
	runs = os.listdir(testDir)
	for counter, run in enumerate(runs):
	  datadir = dataDir + "/" + run
	  datasetHeader, dataset =  getPerfDataset( datadir, configs.NUMBER_OF_COUNTERS )
	  testResults.append(compareDataset( baseDataset, getDatasetArray(dataset)))

  validationResults = []
  if validationDir != None:
	runs = os.listdir(validationDir)
	for counter, run in enumerate(runs):
	  datadir = dataDir + "/" + run
	  datasetHeader, dataset =  getPerfDataset( datadir, configs.NUMBER_OF_COUNTERS )
	  validationResults.append(compareDataset( baseDataset, getDatasetArray(dataset)))
  return results, testResults, validationResults
	  
  
  
def perfAnalyzerMain(outputDir, networkConfig):
  perfTrainDataDir=sys.argv[1]
  perfTestDataDir=sys.argv[2] 
  perfValidDataDir = sys.argv[3]
  #outputDir = sys.argv[4]
  outputFilePrefix = outputDir + "/" + "report."


  #NUMBER_OF_COUNTERS=33
  #NUMBER_OF_COUNTERS=19

  #start_range = 0

  model = None
  logFileName = outputFilePrefix + "log"
  log_file = open(logFileName, 'w')
  
  training_runs = os.listdir(perfTrainDataDir)
  #assert len(training_runs) >= configs.EXPERIMENT_EPOCHS[-1]

  #for epoch in configs.EXPERIMENT_EPOCHS:
  epoch_count = 0
  for (start_range,end_range) in configs.EXPERIMENT_EPOCHS:
	epoch_count += 1	
	#assert len(training_runs) >= end_range ##TODO: runs list have same sequence of dirs in all execution???
	#end_range = epoch
	
	########################
    ## Training
    ########################
    
  	#for run in runs:
  	for i in range(start_range, end_range):
  	  run = training_runs[i]
  	  print "Training with", run
  	  datadir = perfTrainDataDir + "/" + run
  	
  	  datasetHeader, dataset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )
  	  dataArray = getDatasetArray(dataset)
  	  trainingDataset = preprocessDataArray(dataArray)
  
  	  if model == None:
  		model, train_loss  = trainAutoencoder( networkConfig, trainingDataset, trainingDataset, log_file )
  	  else:
  		model, train_loss  = trainAutoencoder( networkConfig, trainingDataset, trainingDataset, log_file, model )
  	
      
	#outFile = open(outputFilePrefix + str(epoch),'w') 
	outFile = open(outputFilePrefix + str(epoch_count)+"."+str(start_range) + "-" + str(end_range),'w') 

	runs = os.listdir(perfTestDataDir)
	
	#print >> outFile, "Actual Positive", len(runs) ## no false sharing
	anomalousRunCount = 0
	for run in runs:
	  print "Testing with", run
	  datadir = perfTestDataDir + "/" + run
	
	  datasetHeader, dataset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )
	  dataArray = getDatasetArray(dataset)
	  dataArray = preprocessDataArray(dataArray)
	  reconstructedData, anomalyCount = runTrainedAutoencoder( model, dataArray, datasetHeader, configs.THRESHOLD_ERROR, outFile )
	  if anomalyCount > 0 :
		print >> outFile, run
		anomalousRunCount += 1
	
	truePositive = anomalousRunCount ## a
	falseNegative = len(runs) - anomalousRunCount ## b
	print >> outFile, "Actual Positive", len(runs)
	print >> outFile, "True Positive", anomalousRunCount
	print >> outFile, "False Negative", len(runs) - anomalousRunCount 
	
	print "Total run ", len(runs) 
	print "Total anomalous run found ", anomalousRunCount
	
	
	runs = os.listdir(perfValidDataDir)
	
	anomalousRunCount = 0
	for run in runs:
	  print "Testing with", run
	  datadir = perfValidDataDir + "/" + run
	
	  datasetHeader, dataset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )
	  dataArray = getDatasetArray(dataset)
	  dataArray = preprocessDataArray(dataArray)
	  reconstructedData, anomalyCount = runTrainedAutoencoder( model, dataArray, datasetHeader, configs.THRESHOLD_ERROR, outFile )
	  if anomalyCount > 0 :
		print >> outFile, run
		anomalousRunCount += 1
	
	trueNegative = len(runs) - anomalousRunCount ## d
	falsePositive = anomalousRunCount  ## c
	print >> outFile, "Actual Negative", len(runs)
	print >> outFile, "True Negative", len(runs) - anomalousRunCount
	print >> outFile, "False Positive", anomalousRunCount 
  
	print "Total run ", len(runs) 
	print "Total anomalous run found ", anomalousRunCount
	##calculate F score 
	
	precision = 0 
	if truePositive+falsePositive != 0:
	  precision = float(truePositive)/(truePositive+falsePositive)
	  
	recall = float(truePositive)/(truePositive+falseNegative)
	fscore = 0
	if precision + recall != 0:
	  fscore = 2* (precision*recall)/(precision+recall) ##harmonic mean of precision and recall

	print >> outFile, "Precision" , precision
	print >> outFile, "Recall", recall
	print >> outFile, "Fscore", fscore
	outFile.close()
	
	
	print "Report: ", outputFilePrefix + str(start_range) + "-" + str(end_range)
	
	#start_range = epoch
  	


if __name__ == "__main__" :

  """
  ##analyzing the dataset
  dataDir = sys.argv[1]
  testDir = None 
  if len(sys.argv) > 2 :
	testDir = sys.argv[2]
  
  if len(sys.argv) > 3 :
	validationDir = sys.argv[3]
  result, testResult, validationResult = analyzeVariationInData(dataDir,  testDir, validationDir)
  plotDataList( result, "variance.png", "variance",testResult, validationResult )
 
  """
  #print configs.NETWORK_CONFIGS
  if len(sys.argv) < 5:
	print "Usage: perfpoint_trainer.py training_dataset_path testing_dataset_path validatin_dataset_path output_path"
	print "Knobs: in configs.py"
	exit()
  outputDir = sys.argv[4]
  for key in configs.NETWORK_CONFIGS:
	print key , configs.NETWORK_CONFIGS[key]
	experimentOutputDirName = outputDir + "/" + key
	mkdir_p(experimentOutputDirName)
	perfAnalyzerMain(experimentOutputDirName, configs.NETWORK_CONFIGS[key])
 
  

