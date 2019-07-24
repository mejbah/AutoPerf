"""
Copyright (c) 2018 Mejbah ul Alam, Justin Gottschlich, Abdullah Muzahid

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

"""
File name: autoperf.py
File description: Main script for training models and test AutoPerf
"""

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


"""
Defintions of tuples used by this script
"""
AnomalyTuple = namedtuple('AnomalyTuple', 'run, sample_count, anomalous_sample_count, ranking')
AccuracyTuple = namedtuple('AccuracyTuple', 'true_positive, false_negative, false_positive, true_negative')


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
    eventID += 1
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

          normalizedCounter = ( currCounter / ( instructionCount * threadCount ) )* configs.SCALE_UP_FACTOR
          if i==0:
            newSample = []
            newSample.append(normalizedCounter)
            dataset.append(newSample)
            dataset[linenumber-3].append(normalizedCounter)

  return datasetHeader, dataset


"""
Convert list to numpy array
"""
def getDatasetArray( dataset ):

  dataArray = np.array(dataset, dtype='float32')
  return dataArray


"""
Error function definition
"""
def getMSE(old, new):
  squaredError = ((old - new) ** 2)
  return np.mean(squaredError)
  #return mean_squared_error(old,new)


"""
Distance calculation between reconstructed and original inputs
"""
def getNormalizedDistance( old, new ):
  dist = np.linalg.norm(old-new)
  origin = np.linalg.norm(old)
  return dist/origin


"""
Function for ranking of anomalies
in : list of tuple of (event,error)
out : ranking and majority voting update
"""
def rankAnomalousPoint( sampleErrorsTuple, rankingMap ):

  sampleErrorsTuple.sort(key=lambda tup : tup[1], reverse=True)
  for index, errorTuple in enumerate(sampleErrorsTuple):
	  rankingMap[errorTuple[0]][index] += 1

  return rankingMap


"""
Report ranking based on majority voting result
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
Calculate reconstruction error : normalized distance
"""
def detectAnomalyPoints( realData, predictedData, datasetHeader, thresholdLoss, outFile=None ):
  datasetLen = realData.shape[0]
  dataLen = realData.shape[1]
  anomalyCount = 0

  """
  map for ranking :
  key = counter name, val = list of len(NUMBER_OF_COUNTERS)
  each pos corresponds to rank, smaller is better
  """
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
      #update ranking
      rankingMap = rankAnomalousPoint(errorList, rankingMap)

  votingResult = reportRanks( rankingMap )

  return datasetLen, anomalyCount, votingResult



"""
Calculate a threshold based on distribution of reconstrucion errors
"""
def getReconstructionErrorThreshold( model, perfTestDataDir, runs ):

  errors = []
  for run in runs:
    print(("Testing with", run))
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
Run inference with dataset for test runs(executions) in the Datadir, write results in outFile
and return number of runs found as anomalous
"""
def testAutoencoder( model, perfTestDataDir, runs, outFile, threshold_error ):
  anomalousRunCount = 0
  for run in runs:
    print(("Testing with", run))
    datadir = perfTestDataDir + "/" + run

    datasetHeader, dataset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )
    dataArray = getDatasetArray(dataset)
    dataArray = preprocessDataArray(dataArray)
    reconstructedData, anomalyCount = runTrainedAutoencoder( model, dataArray, datasetHeader, threshold_error, outFile )
    if anomalyCount > 0 :
	    print("Anomaly found in execution ", run, " in", anomalyCount, " samples", file=outFile)
	    anomalousRunCount += 1

  return anomalousRunCount


"""
Return list of tuple(run,total_sample, anomalous_sample, ranking)
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


"""
Preprocessing of data
"""
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


"""
Util function for aggregating data files from all profile runs
"""
def getTrainDataSequence(dataDir, testDir=None, validationDir=None ):
  runs = os.listdir(dataDir) ##no sequence of directory assigned, it should not matter

  print(("Total execution/directory found for training data:" , len(runs)))
  return runs


"""
Util function for analyzing dataset
"""
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


"""
I/O util for writing log of results
"""
def writeTestLog(logFile, anomalySummary):
  for tuple in anomalySummary:
    print(tuple.run, tuple.sample_count, tuple.anomalous_sample_count, file=logFile)
    ## AnomalyTuple(run = run, sample_count=dataLen, anomalous_sample_count=anomalyCount, ranking = ranking)


"""
Inferencing of model and determination of anomalies
"""
def testModel( model, threshold_error, nonAnomalousDataDir, anomalousDataDir, logFile=None ):
  print("..Testing Non-anomalous")
  negative_runs = os.listdir(nonAnomalousDataDir)
  anomaly_summary = testAnomaly( model, nonAnomalousDataDir, negative_runs, threshold_error )
  if logFile != None:
    print("\n..Testing nonAnomalousData..\n", file=logFile)
    writeTestLog(logFile, anomaly_summary)
  false_positive = len(anomaly_summary)
  true_negative = len(negative_runs) - false_positive
  print("..Testing Anomalous")
  positive_runs = os.listdir(anomalousDataDir)
  anomaly_summary = testAnomaly( model, anomalousDataDir, positive_runs, threshold_error )
  if logFile != None:
    print("\n..Testing AnomalousData..\n", file=logFile)
    writeTestLog(logFile, anomaly_summary)
  true_positive = len(anomaly_summary)
  false_negative = len(positive_runs) - true_positive
  return AccuracyTuple(true_positive=true_positive, false_negative = false_negative, false_positive=false_positive, true_negative=true_negative)


"""
Function for gathering statistics on accuracy of a model for a specific test dataset and validation dataset
"""
def testModelAccuracy( model, outFilename, threshold_error, perfTestDataDir, perfValidDataDir ):

  outFile = open(outFilename, 'w')
  ## test for anomalous dataset
  runs = os.listdir(perfTestDataDir)


  anomalousRunCount = testAutoencoder( model, perfTestDataDir, runs, outFile, threshold_error )

  truePositive = anomalousRunCount ## a
  falseNegative = len(runs) - anomalousRunCount ## b
  print("Actual Positive", len(runs), file=outFile)
  print("True Positive", anomalousRunCount, file=outFile)
  print("False Negative", len(runs) - anomalousRunCount, file=outFile)

  print(("Total run ", len(runs)))
  print(("Total anomalous run found ", anomalousRunCount))


  ##validation with correct(not anomalous)  dataset
  runs = os.listdir(perfValidDataDir)

  anomalousRunCount = testAutoencoder( model, perfValidDataDir, runs, outFile, threshold_error )

  trueNegative = len(runs) - anomalousRunCount ## d
  falsePositive = anomalousRunCount  ## c
  print("Actual Negative", len(runs), file=outFile)
  print("True Negative", len(runs) - anomalousRunCount, file=outFile)
  print("False Positive", anomalousRunCount, file=outFile)

  print(("Total run ", len(runs)))
  print(("Total anomalous run found ", anomalousRunCount))

  ##calculate F score
  precision = 0
  if truePositive+falsePositive != 0:
    precision = float(truePositive)/(truePositive+falsePositive)

  recall = float(truePositive)/(truePositive+falseNegative)
  fscore = 0
  if precision + recall != 0:
    fscore = 2* (precision*recall)/(precision+recall) ##harmonic mean of precision and recall

  print("Precision" , precision, file=outFile)
  print("Recall", recall, file=outFile)
  print("Fscore", fscore, file=outFile)
  outFile.close()


  print(("Report: ", outFilename))



"""
Aggregate all data and train with all data at once
"""

def perfAnalyzerMainTrain( perfTrainDataDir, outputDir, autoencoder, threshold_final=None , saveTrainedNetwork=False):

  model = None

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
    print("Not enough data for training this iteration")
    sys.exit(1)
  else:
    print(("Training dataset size", len(dataset)))
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

  return train_loss_list, validation_loss_list



"""
Train data with one or more exeuction data in one batch
"""
def perfAnalyzerMainTrainSequence( perfTrainDataDir, outputDir, autoencoder, threshold_final=None , saveTrainedNetwork=False):

  model = None

  training_sequence = getTrainDataSequence(perfTrainDataDir)
  #for epoch in configs.EXPERIMENT_EPOCHS:
  train_loss_list = []
  reconstruction_error_list = []
  validation_loss_list = []
  #for train_run in training_sequence:
  i = 0
  while i < len(training_sequence):
    train_run = training_sequence[i]
    print(("Training with", train_run))
    datadir = perfTrainDataDir + "/" + train_run

    datasetHeader, dataset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )


    while i < len(training_sequence):
      print ("small input, adding more data for the batch")
      if(i+1 == len(training_sequence)):
        break ## handle error
      print(("adding ", training_sequence[i+1]))
      datadir = perfTrainDataDir + "/" + training_sequence[i+1]
      i += 1
      redundantHeader, additionalDatatset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )
      dataset.extend(additionalDatatset)
    i += 1
    dataArray = getDatasetArray(dataset)
    trainingDataset = preprocessDataArray(dataArray)

    if len(dataset) < configs.NUMBER_OF_COUNTERS * 2:
      print("Not enough data for training this iteration")
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

  return train_loss_list, validation_loss_list


"""
Util function for determining different topologies
"""
def getRangeOfNode( n ):

  list_of_numbers = []
  for i in range( int(n/4), int(3*n/4) ):
    if i> 5: #TODO: limit ???
      list_of_numbers.append(i)
  return list_of_numbers


"""
Run inference on input data and calculate reconstrution error
@return : list of reconstrution errors
"""
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
    print("Not enough data for training this iteration")
    sys.exit(1)
  else:
    print(("Training dataset size", len(dataset)))
  trainingDataset = getDatasetArray(dataset)
  model, train_loss, validation_loss  = keras_autoencoder.trainAutoencoder( autoencoder, trainingDataset )
  train_loss_list.extend(train_loss)
  validation_loss_list.extend(validation_loss)

  if saveTrainedNetwork == True :
    model.save(outputDir + "/" + configs.MODEL_SAVED_FILE_NAME + "_"+ str(i))

  return model


"""
Calculates threshold of reconstruction errors = mean + 3 * std of reconstruction errors
"""
def calcThresoldError(reconstructionErrors):
  meanVal = np.mean(reconstructionErrors)
  meanVal += ( 3 * np.std(reconstructionErrors))
  return meanVal


"""
Train and test one autoencoder model with the datasets stored in the given directories
"""
def trainAndTest( autoencoder, trainDataDir, nonAnomalousTestDir, anomalousTestDataDir, logFile=None ):
  model = aggregateAndTrain( trainDataDir, autoencoder )

  print("..Training Complete")
  datasetTrainErrors = getReconstructionErrors(trainDataDir, model)
  threshold_error = calcThresoldError(datasetTrainErrors)

  print("..Thresold determined")
  print("Threshold : ", threshold_error, file=logFile)

  test_result = testModel( model, threshold_error, nonAnomalousTestDir, anomalousTestDataDir, logFile )

  model_string = keras_autoencoder.getAutoendoerShape( model )

  return (model_string, threshold_error, test_result)



"""
Create autoencoders from lists of input length and array of layers
"""
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



"""
Train and test provided list of candidate_autoencoders
or serach all possible networks in range of NUMBER_OF_HIDDEN_LAYER_TO_SEARCH
"""
def AutoPerfMain(candidate_autoencoders=None):
  trainDataDir=sys.argv[1]
  nonAnomalousTestDataDir=sys.argv[2]
  anomalousTestDataDir=sys.argv[3]
  outputDir = sys.argv[4]
  mkdir_p(outputDir)

  if candidate_autoencoders == None :
    candidate_autoencoders = getTopologies( configs.NUMBER_OF_COUNTERS, configs.NUMBER_OF_HIDDEN_LAYER_TO_SEARCH )

  out_file = open(outputDir + "/autoperf.out", 'w')
  log_file = open(outputDir + "/autoperf.log",'w')
  print("Number of autoencoder topologies:",  len(candidate_autoencoders), file=log_file)

  print("network, error, true_positive, false_negative, true_negative, false_positive", file=out_file)

  print("Train: ", trainDataDir, file=log_file)
  print("Test(nonAnomalous): ", nonAnomalousTestDataDir, file=log_file)
  print("Test(Anomalous): ", anomalousTestDataDir, file=log_file)
  out_file.close()
  log_file.close()

  for i in range(len(candidate_autoencoders)):
    out_file = open(outputDir + "/autoperf.out", 'a')
    log_file = open(outputDir + "/autoperf.log", 'a')
    print(("..Autoencder topology: ", keras_autoencoder.getAutoendoerShape(candidate_autoencoders[i])))
    print("\n..Autoencder topology: ", keras_autoencoder.getAutoendoerShape(candidate_autoencoders[i]), file=log_file)
    output = trainAndTest( candidate_autoencoders[i], trainDataDir, nonAnomalousTestDataDir, anomalousTestDataDir, log_file )

    ##output -> (model_string, threshold_error, test_result)
    print(output[0], output[1], output[2].true_positive, output[2].false_negative, output[2].true_negative, output[2].false_positive, file=out_file)
    out_file.close()
    log_file.close()


  print(("..Output to file ", outputDir+"/autoperf.out"))
  print(("..Log file ", outputDir+"/autoperf.log"))




if __name__ == "__main__" :

  if(len(sys.argv) < 4):
    print("Usage: autoperf.py path/to/trainingdata path/to/noAnomalousTestData path/to/anomalousTestData path/to/output")
    sys.exit()

  input_dim = configs.NUMBER_OF_COUNTERS
  #hidden_dims = [[ 16, 8 ], [16], [8], [16,8,4]]
  #encoding_dims = [4, 8, 4, 2]
  hidden_dims = [[ 16 ]]
  encoding_dims = [8]


  candidate_autoencoders = []
  for hidden_dim, encoding_dim in zip(hidden_dims, encoding_dims):
    autoencoder = keras_autoencoder.getAutoencoder(input_dim,  encoding_dim, hidden_dim)
    candidate_autoencoders.append(autoencoder)
  AutoPerfMain(candidate_autoencoders)
  #AutoPerfMain()
