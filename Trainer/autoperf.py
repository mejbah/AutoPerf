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
from collections import namedtuple

#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.metrics import mean_squared_error

AnomalyTuple = namedtuple('AnomalyTuple', 'run, sample_count, anomalous_sample_count, ranking')
AccuracyTuple = namedtuple('AccuracyTuple', 'true_positive, false_negative, false_positive, true_negative')

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
  eventID = 0 
  for i in range(0, numberOfCounters):
    #if i==2 or i==15 or  i==16  : 
    #  continue #TODO: 2 counters are not set in PAPI, temp fix , remove this once problem is resolved
    
    filename = dirName + "/event_" + str(eventID) + "_perf_data.csv"
    #while not os.path.isfile(filename):
    while not os.path.isfile(filename) or eventID==15 or eventID==16: #TODO: only for mysql, remove this for others
      assert eventID < configs.MAX_COUNTERS
      eventID += 1
      filename = dirName + "/event_" + str(eventID) + "_perf_data.csv"
    
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
          if i==0:
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
def detectAnomalyPoints( realData, predictedData, datasetHeader, thresholdLoss, outFile=None ):
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
  
  return datasetLen, anomalyCount, votingResult

 


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
return list of tuple(run,total_sample, anomalous_sample, ranking)
"""
def testAnomaly( model, testDataDir, runs, threshold_error ):
  test_summary = []
  for run in runs:
    datadir = testDataDir + "/" + run
    datasetHeader, dataset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )
    dataArray = getDatasetArray(dataset)
    dataArray = preprocessDataArray(dataArray)
    decoded_data = keras_autoencoder.predict( model, dataArray ) 
    datasetLen, anomalyCount, ranking = detectAnomalyPoints(dataArray, decoded_data, datasetHeader, threshold_error)
    if anomalyCount > datasetLen * configs.PERCENT_SAMPLE_FOR_ANOMALY : ##TODO: use a thresold, small % of anomaly can me ignored
      test_summary.append(AnomalyTuple(run = run, sample_count=datasetLen, anomalous_sample_count=anomalyCount, ranking = ranking))
  return test_summary


"""
Run trained autoencoder and detect anomalous samples based on 'thresoldLoss' 
and write output in 'outFile'
"""
def runTrainedAutoencoder( model, testDataArray, datasetHeader, thresholdLoss, outFile ):
  
  decoded_data = keras_autoencoder.predict( model, testDataArray ) 
  dataLen, anomalyCount, ranking = detectAnomalyPoints(testDataArray, decoded_data, datasetHeader, thresholdLoss)
  
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

def writeTestLog(logFile, anomalySummary):
  for tuple in anomalySummary:
    print >> logFile, tuple.run, tuple.sample_count, tuple.anomalous_sample_count
    ## AnomalyTuple(run = run, sample_count=dataLen, anomalous_sample_count=anomalyCount, ranking = ranking)

def testModel( model, threshold_error, nonAnomalousDataDir, anomalousDataDir, logFile=None ):
  print "..Testing Non-anomalous"
  negative_runs = os.listdir(nonAnomalousDataDir)
  anomaly_summary = testAnomaly( model, nonAnomalousDataDir, negative_runs, threshold_error )
  if logFile != None:
    print >> logFile, "\n..Testing nonAnomalousData..\n"
    writeTestLog(logFile, anomaly_summary)
  false_positive = len(anomaly_summary)
  true_negative = len(negative_runs) - false_positive
  print "..Testing Anomalous"
  positive_runs = os.listdir(anomalousDataDir)
  anomaly_summary = testAnomaly( model, anomalousDataDir, positive_runs, threshold_error )
  if logFile != None:
    print >> logFile, "\n..Testing AnomalousData..\n"
    writeTestLog(logFile, anomaly_summary)
  true_positive = len(anomaly_summary)
  false_negative = len(positive_runs) - true_positive 
  return AccuracyTuple(true_positive=true_positive, false_negative = false_negative, false_positive=false_positive, true_negative=true_negative)

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
    model, train_loss, validation_loss  = keras_autoencoder.trainAutoencoder( autoencoder, trainingDataset ) 
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
  for i in range( int(n/4), int(3*n/4) ):
    if i> 5: #TODO: limit ???
      list_of_numbers.append(i)
  return list_of_numbers


def getReconstructionErrors( perfTestDataDir, model ):

  runs = os.listdir(perfTestDataDir) 
  reconstructErrorList = []
  for run in runs:
    #print ("Reconstruction of execution ", run)
    datadir = perfTestDataDir + "/" + run
  
    datasetHeader, dataset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )
    dataArray = getDatasetArray(dataset)
    dataArray = preprocessDataArray(dataArray)
     
    datasetLen = dataArray.shape[0]
    decodedData = keras_autoencoder.predict( model, dataArray )  
    for x  in range(datasetLen):
      #reconstructionError = getNormalizedDistance( realData[x], predictedData[x] )
      reconstructionError = getMSE( dataArray[x], decodedData[x] )
      #for debugging
      reconstructErrorList.append(reconstructionError)
  return reconstructErrorList



"""
Aggregate all data and train with all data at once
@return trained model
"""
  
def aggregateAndTrain( perfTrainDataDir, autoencoder, saveTrainedNetwork=False,outputDir=None ):
  
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
  model, train_loss, validation_loss  = keras_autoencoder.trainAutoencoder( autoencoder, trainingDataset ) 
  train_loss_list.extend(train_loss)
  validation_loss_list.extend(validation_loss)

  if saveTrainedNetwork == True :
    model.save(outputDir + "/" + configs.MODEL_SAVED_FILE_NAME + "_"+ str(i))

  return model

"""
threshold = mean + 3 * std of reconstruction errors
"""
def calcThresoldError(reconstructionErrors):
  meanVal = np.mean(reconstructionErrors)
  meanVal += ( 3 * np.std(reconstructionErrors))
  return meanVal
  
 
def trainAndTest( autoencoder, trainDataDir, nonAnomalousTestDir, anomalousTestDataDir, logFile=None ):
  model = aggregateAndTrain( trainDataDir, autoencoder )

  print "..Training Complete" 
  datasetTrainErrors = getReconstructionErrors(trainDataDir, model)
  threshold_error = calcThresoldError(datasetTrainErrors)
  print "..Thresold determined"
  
  test_result = testModel( model, threshold_error, nonAnomalousTestDir, anomalousTestDataDir, logFile )

  model_string = keras_autoencoder.getAutoendoerShape( model )

  return (model_string, threshold_error, test_result)

"""
Train and test with given autoencoder network
Outputs reconstruction errors for train and test dataset
"""
def runAutoencoder( inputDim, encodeDim, middleLayers, perfTrainDataDir, perfTestDataDir, outputDir ):
  mkdir_p(outputDir) 
  candidate_autoencoder = keras_autoencoder.getAutoencoder(input_dim, encode_dim, hidden_dims)
  autoencoder = keras_autoencoder.getAutoencoder(inputDim, encodeDim, middleLayers)
  training_losses, validation_losses = perfAnalyzerMainTrain( perfTrainDataDir, outputDir, autoencoder )

  datasetTrainErrors = getReconstructionErrors(perfTrainDataDir,autoencoder)
  datasetTestErrors = getReconstructionErrors(perfTestDataDir, autoencoder)
  
    
  outFilename = outputDir + "/reconstruction_errors_train.pdf"
  plotDataList( datasetTrainErrors, outFilename)

  outFilename = outputDir + "/reconstruction_errors_train.out"
  outFile = open(outFilename, 'w')
  for val in datasetTrainErrors:
    print >> outFile, val

  outFile.close()
   
  outFilename = outputDir + "/reconstruction_errors_test.pdf"
  plotDataList( datasetTestErrors, outFilename)
  outFilename = outputDir + "/reconstruction_errors_test.out"
  outFile = open(outFilename, 'w')
  for val in datasetTestErrors:
    print >> outFile, val

  outFile.close()



def getTopologies( inputLen, numberOfLayers ):
  
  candidate_autoencoders = []
  prefix = []
  for layer in range(numberOfLayers):
    number_of_prefix = 1
    if layer > 0:
      number_of_prefix = len(prefix)
    for prefix_layer in range(number_of_prefix):
      curr_layer_nodes = 1
      if layer == 0:
        curr_layer_nodes = getRangeOfNode(inputLen)
      else:
        curr_layer_nodes = getRangeOfNode(prefix[prefix_layer][-1])
      for n in curr_layer_nodes:
        new_prefix = []
        if layer > 0:
          for nodes in prefix[prefix_layer]:
            new_prefix.append(nodes)

        
        #print "curr network: ", inputLen, new_prefix, n
        autoencoder = keras_autoencoder.getAutoencoder(inputLen, n, new_prefix)
        candidate_autoencoders.append(autoencoder)
        new_prefix.append(n) #add new layer nodes
        prefix.append(new_prefix) #store prefix in this layer

      
    prefix = prefix[number_of_prefix-1:]

  return candidate_autoencoders
     

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



"""
train and test given single autoencoder configuration
"""
def iAmFeelingLucky( input_dim, hidden_dims, encode_dim ):
  trainDataDir=sys.argv[1]
  nonAnomalousTestDataDir=sys.argv[2] 
  anomalousTestDataDir=sys.argv[3] 
  outputDir = sys.argv[4]
  mkdir_p(outputDir)
  candidate_autoencoder = keras_autoencoder.getAutoencoder(input_dim, encode_dim, hidden_dims)
  out_file = open(outputDir + "/autoperf.out", 'a')
  log_file = open(outputDir + "/autoperf.log", 'a')
  if os.stat(outputDir + "/autoperf.out").st_size == 0:
    print >> out_file, "network - error - true_positive - false_negative - true_negative - false_positive"
  print >> log_file, "Train: ", trainDataDir
  print >> log_file, "Test(nonAnomalous): ", nonAnomalousTestDataDir
  print >> log_file, "Test(Anomalous): ", anomalousTestDataDir
  print "..Autoencder topology: ", keras_autoencoder.getAutoendoerShape(candidate_autoencoder)
  print >> log_file, "\n..Autoencder topology: ", keras_autoencoder.getAutoendoerShape(candidate_autoencoder)
  output = trainAndTest( candidate_autoencoder, trainDataDir, nonAnomalousTestDataDir, anomalousTestDataDir, log_file )
  ##output -> (model_string, threshold_error, test_result)
  print >> out_file, output[0], "-", output[1], "-", output[2].true_positive, "-", output[2].false_negative, "-", output[2].true_negative, "-", output[2].false_positive
  out_file.close()  
  log_file.close()


def AutoPerfMain():
  trainDataDir=sys.argv[1]
  nonAnomalousTestDataDir=sys.argv[2] 
  anomalousTestDataDir=sys.argv[3] 
  outputDir = sys.argv[4]
  mkdir_p(outputDir)
  candidate_autoencoders = getTopologies( configs.NUMBER_OF_COUNTERS, configs.NUMBER_OF_HIDDEN_LAYER_TO_SEARCH )
  print len(candidate_autoencoders)

  out_file = open(outputDir + "/autoperf.out", 'w')
  log_file = open(outputDir + "/autoperf.log",'w')
  print >> out_file, "network, error, true_positive, false_negative, true_negative, false_positive"
  print >> log_file, "Train: ", trainDataDir
  print >> log_file, "Test(nonAnomalous): ", nonAnomalousTestDataDir
  print >> log_file, "Test(Anomalous): ", anomalousTestDataDir
  out_file.close()  
  log_file.close()
 
  for i in range(len(candidate_autoencoders)): 
    out_file = open(outputDir + "/autoperf.out", 'a')
    log_file = open(outputDir + "/autoperf.log", 'a')
    print "..Autoencder topology: ", keras_autoencoder.getAutoendoerShape(candidate_autoencoders[i])
    print >> log_file, "\n..Autoencder topology: ", keras_autoencoder.getAutoendoerShape(candidate_autoencoders[i])
    output = trainAndTest( candidate_autoencoders[i], trainDataDir, nonAnomalousTestDataDir, anomalousTestDataDir, log_file )
    ##output -> (model_string, threshold_error, test_result)
    print >> out_file, output[0], output[1], output[2].true_positive, output[2].false_negative, output[2].true_negative, output[2].false_positive
    out_file.close()  
    log_file.close()


  print "..Output to file ", outputDir+"/autoperf.out"
  print "..Log file ", outputDir+"/autoperf.log"

  #for model in candidate_autoencoders:
  #  print keras_autoencoder.getAutoendoerShape( model )  









if __name__ == "__main__" :
 
    
  if(len(sys.argv) == 2 and sys.argv[1] == "test"):
    print ("Running Unit Test")
    unitTest()
    sys.exit()

  if(len(sys.argv) < 4):
    print "Usage: autoperf.py path/to/trainingdata path/to/noAnomalousTestData path/to/anomalousTestData path/to/output"
    sys.exit()


  
  #AutoPerfMain()

  #or
  ## test known best network
  input_dim = 16
  #hidden_dims_list = [ [ 22, 16, 12], [ 23, 16, 12], [ 24, 16, 12] ]
  #hidden_dims_list = [ [ 22, 16], [ 23, 16], [ 24, 16] ]
  #hidden_dims_list = [ [ 24, 16, 8], [ 25, 16, 8], [ 26, 16, 8] ]
  #hidden_dims_list = [ [ 28, 24, 10], [24, 20, 16] ]
  #hidden_dims_list = [ [ 28, 24, 16], [24, 24, 20] ]
  hidden_dims_list = [ [ 12, 8 ] ]
  encode_dim = 4
  #encode_dim = 8
  #encode_dim = 4
  for hidden_dims in hidden_dims_list:
    #iAmFeelingLucky( input_dim, hidden_dims, encode_dim )
    # for paper figure data collection
    runAutoencoder( input_dim, encode_dim, hidden_dims, sys.argv[1], sys.argv[2], sys.argv[3] );

  #perfTrainDataDir=sys.argv[1]
  #perfTestDataDir=sys.argv[2] 
  #outputDir = sys.argv[3]
  #mkdir_p(outputDir)

  ##set network configs
  #inputLen = configs.NUMBER_OF_COUNTERS
  #numberOfLayers = configs.NUMBER_OF_HIDDEN_LAYER_TO_SEARCH

  #bestNetwork, trainedAutoencoder, minLoss = findBestNetwork(inputLen, numberOfLayers, perfTrainDataDir, outputDir)
  #testModelAccuracy( trainedAutoencoder, outputDir+"/accuracy.out", minLoss, perfTestDataDir, perfTrainDataDir)

