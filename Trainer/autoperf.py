#!/home/mejbah/tools/python2.7/bin/python2.7
from __future__ import division
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg') #instead of Xserver for png
import matplotlib.pyplot as plt
import pprint
import numpy as np
import plot_utils
import os
import errno
import configs
from utils import *
from random import shuffle
import keras_autoencoder
import copy
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.metrics import mean_squared_error



"""
Returns number of executions profile csv file present in the directory
"""
def getNumberOfExecProfile( dirName ):
  profiles = [ file.endswith(".csv") for file in dirName]
  return len(profiles)

def getExecProfileFileNames( dirName ):
  profiles = [ file.endswith(".csv") for file in dirName]
  return len(profiles)


"""
Create list of sample with all the counter values from profile data
"""
def getPerfDataset( dirName , numberOfCounters ):
  datasetHeader = []
  dataset = []
   
  for i in range(1, numberOfCounters+1+1):
    if i==2 or i==15 or  i==16  : 
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
        
          #normalizedCounter = ( currCounter / ( instructionCount * threadCount ) )* configs.SCALE_UP_FACTOR
          #normalizedCounter = ( currCounter / ( instructionCount ) )* configs.SCALE_UP_FACTOR
          normalizedCounter =  currCounter #TODO: fix this, changed for zero mean data processing
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

		
"""
Convert list to numpy arra
"""
def getDatasetArray( dataset ):
  
  dataArray = np.array(dataset, dtype='float32')
  return dataArray


def getMSE(old, new):
  squaredError = ((old - new) ** 2)
  return np.mean(squaredError)
  #return mean_squared_error(old,new)

def getNormalizedDistance( old, new ):
  dist = np.linalg.norm(old-new)
  origin = np.linalg.norm(old)
  return dist/origin


"""
in : list of tuple of (event,error)
out : ranking and majority voting update
"""
def rankAnomalousPoint( sampleErrorsTuple, rankingMap ):
  
  sampleErrorsTuple.sort(key=lambda tup : tup[1], reverse=True)
  for index, errorTuple in enumerate(sampleErrorsTuple):
	  rankingMap[errorTuple[0]][index] += 1	
	
  return rankingMap 



"""report majority voting result
in : rankingMap -- key: counter_name, val: vote count for each pos/rank
out: list of rank for the entire execution anomaly root cause
"""
def reportRanks( rankingMap ) :
  
  anomalousCounterRank = []
  for i in range(configs.NUMBER_OF_COUNTERS): ##for each possible ranks
    maxVote = 0
    maxkey = "None"
    for key in rankingMap:
      if rankingMap[key][i] > maxVote:
        maxVote = rankingMap[key][i]
        maxkey = key

    anomalousCounterRank.append(maxkey)

  return anomalousCounterRank


"""
calculate reconstruction error : normalized distance 
"""
def detectAnomalyPoints( realData, predictedData, outFile, datasetHeader, thresholdLoss ):
  datasetLen = realData.shape[0]
  dataLen = realData.shape[1]
  anomalyCount = 0


  ############################################################################################
  #map for ranking : key = counter name, val = list of len(NUMBER_OF_COUNTERS) 
  #											  each pos corresponds to rank, smaller is better
  #############################################################################################
  rankingMap = {}
  for counterName in datasetHeader:
    rankingMap[counterName] = [0] * configs.NUMBER_OF_COUNTERS
 
  reconstructErrorList = [] 
  for x  in range(datasetLen):
    #reconstructionError = getNormalizedDistance( realData[x], predictedData[x] )
    reconstructionError = getMSE( realData[x], predictedData[x] )
    #for debugging
    reconstructErrorList.append(reconstructionError)	
    #end for debugging
    #if(reconstructionError > configs.THRESHOLD_ERROR):
    if(reconstructionError > thresholdLoss):
      anomalyCount += 1
      #outputStr = "[" + str(x) + ":" + str(reconstructionError)  + "] " #sample number starting from 0
      errorList = [] #for ranking
      for y in range(0, dataLen):
        dist=abs(predictedData[x][y]-realData[x][y]) / realData[x][y]
        #collect errors and counter tuple
        errorList.append( (datasetHeader[y], dist) )
      #if dist > thresholdLoss:
      #  outputStr += datasetHeader[y]
      #  outputStr += ":"
      #  outputStr += str(dist)
      #  outputStr += " "
      #outFile.write(outputStr)
      #outFile.write("\n")
      #update ranking
      rankingMap = rankAnomalousPoint(errorList, rankingMap)

#	plotDataList(reconstructErrorList, "true_errors.png", "reconstruction error")
  
  votingResult = reportRanks( rankingMap )
  
  return dataLen, anomalyCount, votingResult

 


def getReconstructionErrorThreshold( model, perfTestDataDir, runs ):
 
  errors = []
  for run in runs:
    print("Testing with", run)
    datadir = perfTestDataDir + "/" + run
    datasetHeader, dataset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )
    dataArray = getDatasetArray(dataset)
    realData = preprocessDataArray(dataArray)
    predictedData = model.predict(realData)
    reconstructErrorList = [] 
    datasetLen = realData.shape[0]
    for x in range(datasetLen):
      #reconstructionError = getNormalizedDistance( realData[x], predictedData[x] )
      reconstructionError = getMSE( realData[x], predictedData[x] )
      reconstructErrorList.append(reconstructionError)
  
  errors.append(np.mean(reconstructErrorList) + np.std(reconstructErrorList))


  return (np.mean(errors) + np.std(errors))


"""
test runs(executions) in the Datadir
write outFile
return number of run found as anomalous
"""
def testAutoencoder( model, perfTestDataDir, runs, outFile, threshold_error ):
  anomalousRunCount = 0
  for run in runs:
    print ("Testing with", run)
    datadir = perfTestDataDir + "/" + run
  
    datasetHeader, dataset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )
    dataArray = getDatasetArray(dataset)
    dataArray = preprocessDataArray(dataArray)
    reconstructedData, anomalyCount = runTrainedAutoencoder( model, dataArray, datasetHeader, threshold_error, outFile )
    if anomalyCount > 0 :
	    print >> outFile, "Anomaly found in execution ", run, " in", anomalyCount, " samples"
	    anomalousRunCount += 1
    
  return anomalousRunCount


"""
Run trained autoencoder and detect anomalous samples based on 'thresoldLoss' 
and write output in 'outFile'
"""
def runTrainedAutoencoder( model, testDataArray, datasetHeader, thresholdLoss, outFile ):
  
  #print(model.score(testDataArray))
  
  decoded_data = keras_autoencoder.predict( model, testDataArray ) 
  
  dataLen, anomalyCount, ranking = detectAnomalyPoints(testDataArray, decoded_data, outFile, datasetHeader, thresholdLoss)

  print >>  outFile, ranking

  ##debug print end

  
  
  return decoded_data, anomalyCount


def createDataArray( datapath, numberofexec ):

  topLevelDirName = datapath + "/outputs"
  
  print ("reading dataset\n")
  dataset, datasetHeader = createDataset(numberofexec, topLevelDirName)

  print ("dataset created\n" )  
  #make np array
  dataArray = np.array(dataset, dtype='float32')

  return dataArray , datasetHeader
 
  

def preprocessDataArray( dataset ):
  #zero centering
  #mean_vector = np.mean(dataset, axis=0)
  #processed_dataset = dataset - mean_vector

  #normalize
  #dataset /= np.std(dataset, axis=0)

  ##normalize with max in each column : not working some 0 values?? why not avoid zeros
  datasetMax = np.max(dataset, axis=0)
  processed_dataset = np.nan_to_num(np.true_divide(dataset,datasetMax))

  return processed_dataset


def getTrainDataSequence(dataDir, testDir=None, validationDir=None ):
  runs = os.listdir(dataDir) ##no sequence of directory assigned, it should not matter
  
  print "Total execution/directory found for training data:" , len(runs)
  return runs	


def analyzeVariationInData( dataDir, testDir=None, validationDir=None ):
  
  runs = os.listdir(dataDir)
  results = []
  baseDataset = None
  for counter, run in enumerate(runs):
    datadir = dataDir + "/" + run
    datasetHeader, dataset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )
    if counter == 0:
      baseDataset = getDatasetArray(dataset)
      results.append( (run,0)) 
    else:
      if counter < 50:
        #results.append(compareDataset(baseDataset, getDatasetArray(dataset)))
        results.append(( run, compareDataset(baseDataset, getDatasetArray(dataset))))

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

  sorted_results = sorted(results, key=lambda tup: tup[1], reverse=True)
  return sorted_results, testResults, validationResults
  
 

def testModelAccuracy( model, outFilename, threshold_error, perfTestDataDir, perfValidDataDir ):
  
  outFile = open(outFilename, 'w')
  ## test for anomalous dataset
  runs = os.listdir(perfTestDataDir)

  
  anomalousRunCount = testAutoencoder( model, perfTestDataDir, runs, outFile, threshold_error ) 	
    
  truePositive = anomalousRunCount ## a
  falseNegative = len(runs) - anomalousRunCount ## b
  print >> outFile, "Actual Positive", len(runs)
  print >> outFile, "True Positive", anomalousRunCount
  print >> outFile, "False Negative", len(runs) - anomalousRunCount 
  
  print ("Total run ", len(runs))
  print ("Total anomalous run found ", anomalousRunCount)
  

  ##validation with correct(not anomalous)  dataset	
  runs = os.listdir(perfValidDataDir)

  anomalousRunCount = testAutoencoder( model, perfValidDataDir, runs, outFile, threshold_error ) 	
  
  trueNegative = len(runs) - anomalousRunCount ## d
  falsePositive = anomalousRunCount  ## c
  print >> outFile, "Actual Negative", len(runs)
  print >> outFile, "True Negative", len(runs) - anomalousRunCount
  print >> outFile, "False Positive", anomalousRunCount 

  print ("Total run ", len(runs))
  print ("Total anomalous run found ", anomalousRunCount)

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
  
  
  print ("Report: ", outFilename)



"""
Aggregate all data and train with all data at once
"""
  
def perfAnalyzerMainTrain( perfTrainDataDir, outputDir, autoencoder, threshold_final=None , saveTrainedNetwork=False):

  #outputFilePrefix = outputDir + "/" + "report."
  model = None
  #logFileName = outputFilePrefix + "log"
  #log_file = open(logFileName, 'w')
  
  training_sequence = getTrainDataSequence(perfTrainDataDir)
  train_loss_list = []
  reconstruction_error_list = []
  validation_loss_list = []
  dataset  = []
  for train_run in training_sequence:
    datadir = perfTrainDataDir + "/" + train_run
    redundantHeader, additionalDatatset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )
    processed_dataset_array = preprocessDataArray(getDatasetArray(additionalDatatset))
    dataset.extend(processed_dataset_array.tolist())
  
  if len(dataset) < configs.NUMBER_OF_COUNTERS * 2:
    print "Not enough data for training this iteration" 
    sys.exit(1)
  else:
    print "Training dataset size", len(dataset)
  trainingDataset = getDatasetArray(dataset)
  if model == None:
    model, train_loss, validation_loss  = keras_autoencoder.trainAutoencoder( autoencoder, trainingDataset) 
    train_loss_list.extend(train_loss)
    validation_loss_list.extend(validation_loss)
  else:
    model, train_loss, validation_loss  = keras_autoencoder.trainAutoencoder( model, trainingDataset )
    train_loss_list.extend(train_loss)
    validation_loss_list.extend(validation_loss)
  if saveTrainedNetwork == True :
    model.save(outputDir + "/" + configs.MODEL_SAVED_FILE_NAME + "_"+ str(i))

  ##instead of calculating reconstruction error, used validation loss
  """
  if threshold_final !=None:
    testModelAccuracy( model, outputDir+"/report.accuracy_"+ str(i), threshold_final )
  else:
    reconstruction_error_list.append(getReconstructionErrorThreshold( model, perfTrainDataDir, training_sequence )) 
  return train_loss_list, reconstruction_error_list
  """
  return train_loss_list, validation_loss_list

"""
train data with one or more exeuction data in one batch
"""
def perfAnalyzerMainTrainSequence( perfTrainDataDir, outputDir, autoencoder, threshold_final=None , saveTrainedNetwork=False):

  #outputFilePrefix = outputDir + "/" + "report."
  model = None
  #logFileName = outputFilePrefix + "log"
  #log_file = open(logFileName, 'w')
  
  
  training_sequence = getTrainDataSequence(perfTrainDataDir)
  #for epoch in configs.EXPERIMENT_EPOCHS:
  train_loss_list = []
  reconstruction_error_list = []
  validation_loss_list = []
  #for train_run in training_sequence:
  i = 0
  while i < len(training_sequence):
    train_run = training_sequence[i]
    print ("Training with", train_run)
    datadir = perfTrainDataDir + "/" + train_run
  
    datasetHeader, dataset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )


    #while len(dataset)  < configs.NUMBER_OF_COUNTERS * 2: ##TODO:change should be reversed
    while i < len(training_sequence):
      print ("small input, adding more data for the batch")
      if(i+1 == len(training_sequence)):
        break ## handle error
      print ("adding ", training_sequence[i+1])
      datadir = perfTrainDataDir + "/" + training_sequence[i+1] 
      i += 1
      redundantHeader, additionalDatatset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )
      dataset.extend(additionalDatatset)
    i += 1 
    dataArray = getDatasetArray(dataset)
    trainingDataset = preprocessDataArray(dataArray)
  
    if len(dataset) < configs.NUMBER_OF_COUNTERS * 2:
      print "Not enough data for training this iteration" 
      break
    if model == None:
      model, train_loss, validation_loss  = keras_autoencoder.trainAutoencoder( autoencoder, trainingDataset) 
      train_loss_list.extend(train_loss)
      validation_loss_list.extend(validation_loss)
    else:
      model, train_loss, validation_loss  = keras_autoencoder.trainAutoencoder( model, trainingDataset )
      train_loss_list.extend(train_loss)
      validation_loss_list.extend(validation_loss)
  if saveTrainedNetwork == True :
    model.save(outputDir + "/" + configs.MODEL_SAVED_FILE_NAME + "_"+ str(i))

  ##instead of calculating reconstruction error, used validation loss
  """
  if threshold_final !=None:
    testModelAccuracy( model, outputDir+"/report.accuracy_"+ str(i), threshold_final )
  else:
    reconstruction_error_list.append(getReconstructionErrorThreshold( model, perfTrainDataDir, training_sequence )) 
  return train_loss_list, reconstruction_error_list
  """
  return train_loss_list, validation_loss_list

    
  


def unitTest():
  sampleErrorsTuple = [("y",1), ("x",4)]
  rankingMap = {}
  rankingMap["x"] = [0] * 2
  rankingMap["y"] = [0] * 2
  for i in range(10):
    rankingMap = rankAnomalousPoint( sampleErrorsTuple, rankingMap )
  print (rankingMap)

  
  votingResult = reportRanks( rankingMap )
  print (votingResult)


def getRangeOfNode( n ):
  list_of_numbers = []
  for i in range( int(n/2), n ):
    if i>1:
      list_of_numbers.append(i)
  return list_of_numbers

def findBestNetwork(inputLen, numberOfLayers, dataDir, outputDir):

  print "Input layer len:", inputLen

  ## list for storing all the configuration
  ## tuple ( network_desc, eval )
  ## network_desc = tuple ( [hidden], encoding_dim )
  ## eval = tuple ( training_error, validation_error, reconstruction_thresold )
  networkEvaluations=[] 
  bestNetwork = None
  minLoss = -1
  trainedAutoencoder = None
  ## 1 layer is just input and encoding dim
  range_of_nodes = getRangeOfNode(inputLen) 
  hidden_layer_prefixes = []
  for n in range_of_nodes: ##no hidden layer between input->encoding_layer
    currNetwork = ( inputLen, [], n )
    print "----", currNetwork , "----"
    autoencoder = keras_autoencoder.getAutoencoder(inputLen, n)
    training_losses, validation_losses = perfAnalyzerMainTrain( perfTrainDataDir, outputDir, autoencoder )
    networkEvaluations.append( (currNetwork, training_losses[-1], validation_losses[-1]) )
    if bestNetwork == None or minLoss > validation_losses[-1]:
      minLoss = validation_losses[-1]
      bestNetwork = currNetwork
      trainedAutoencoder = autoencoder
    #update hidden_layer for next layer network construction
    hidden_layer_prefixes.append([n])
  
  for layer in range(1,numberOfLayers):
    newLayerPrefixes = []
    for prefix in hidden_layer_prefixes:
      currNodes = getRangeOfNode(prefix[-1])
      for n in currNodes:
        currNetwork = ( inputLen, prefix, n )
        print "----", currNetwork , "----"
        autoencoder = keras_autoencoder.getAutoencoder(inputLen,n, prefix)
        training_losses, validation_losses = perfAnalyzerMainTrain( perfTrainDataDir, outputDir, autoencoder )
        networkEvaluations.append( (currNetwork, training_losses[-1], validation_losses[-1]) )
        if bestNetwork == None or minLoss > validation_losses[-1]:
          minLoss = validation_losses[-1]
          bestNetwork = currNetwork
          trainedAutoencoder = autoencoder
        
        newPrefix = copy.deepcopy(prefix)
        newPrefix.append(n) 
        newLayerPrefixes.append(newPrefix)
      
    hidden_layer_prefixes = newLayerPrefixes  
    
  print "Network with minimum loss", bestNetwork, " --> loss: ", minLoss
 
  ##write log  
  networkLogFile = open(outputDir + "/network_training.log", 'w')
  for tup in networkEvaluations:
    currNetwork = tup[0]
    trainLoss = tup[1]
    valLoss = tup[2]
    print >> networkLogFile, currNetwork, trainLoss, valLoss

   
  print >> networkLogFile, "Network with minimum loss", bestNetwork, " --> loss: ", minLoss

  
  networkLogFile.close()

  return bestNetwork, trainedAutoencoder, minLoss

  

if __name__ == "__main__" :
 
  if(len(sys.argv)==1):
    print "Usage: autoperf.py path/to/trainingdata path/to/testdata path/to/output"
    sys.exit()
  if(len(sys.argv) == 2):
    print ("Running Unit Test")
    unitTest()
    sys.exit()

  perfTrainDataDir=sys.argv[1]
  perfTestDataDir=sys.argv[2] 
  outputDir = sys.argv[3]
  mkdir_p(outputDir)
  
  ##set network configs
  inputLen = configs.NUMBER_OF_COUNTERS
  numberOfLayers = configs.NUMBER_OF_HIDDEN_LAYER_TO_SEARCH

  bestNetwork, trainedAutoencoder, minLoss = findBestNetwork(inputLen, numberOfLayers, perfTrainDataDir, outputDir)
  testModelAccuracy( trainedAutoencoder, outputDir+"/accuracy.out", minLoss, perfTestDataDir, perfTrainDataDir)


