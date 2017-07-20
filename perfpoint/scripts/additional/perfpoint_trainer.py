#!/home/mejbah/tools/python2.7/bin/python2.7
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
from random import shuffle
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.metrics import mean_squared_error
## log results ###
#climate.disable_default_logging()



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
	
  for i in range(1, numberOfCounters+1):
	  #if i==15 or  i==16  : 
	  #  continue #TODO: 2 counters are not set in PAPI, temp fix , remove this once problem is resolved
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
		      #normalizedCounter = ( currCounter / ( instructionCount * threadCount ) ) #* configs.SCALE_UP_FACTOR
		      normalizedCounter = ( currCounter ) # / ( instructionCount * threadCount ) ) #* configs.SCALE_UP_FACTOR
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

def getMAE(old, new):
  absError = np.fabs(old - new)
  return np.mean(abs)


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


"""
report majority voting result
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
	  reconstructErrorList.append(reconstructionError)	
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
	  model = theanets.Autoencoder(networkConfig, loss='mse')  
	  if logFile != None:
	    print >> logFile, networkConfig 
  # optional: set up additional losses.
  #model.add_loss('mae', weight=0.1)
  #model.set_loss('mse')
  print "Training autoencoder with dateset " , trainingDataArray.shape[0] , "x" , trainingDataArray.shape[1]
  train_loss = 0
  #for train, valid in model.itertrain(trainingDataArray, validationDataArray):
  #train, valid = model.train(trainingDataArray, validationDataArray, algo='layerwise')
  train, valid = model.train(trainingDataArray, validationDataArray, algo='adadelta', learning_rate=0.001)
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


def getReconstructionErrorThreshold( model, perfTestDataDir, runs ):
 
  errors = []
  for run in runs:
    print "Testing with", run
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
  
    #errors.append(np.mean(reconstructErrorList) + np.std(reconstructErrorList))#TODO: am I doing it right?
    errors.append(np.max(reconstructErrorList))


  return (np.mean(errors) + np.std(errors))
  #return np.max(errors)
"""
test runs(executions) in the Datadir
write outFile
return number of run found as anomalous
"""
def testAutoencoder( model, perfTestDataDir, runs, outFile, threshold_error ):
  anomalousRunCount = 0
  print "testAutencoder using dataset:", perfTestDataDir
  for run in runs:
    print "Testing with", run
    datadir = perfTestDataDir + "/" + run
  
    datasetHeader, dataset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )
    dataArray = getDatasetArray(dataset)
    dataArray = preprocessDataArray(dataArray)
    reconstructedData, anomalyCount = runTrainedAutoencoder( model, dataArray, datasetHeader, threshold_error, outFile )
    if anomalyCount > 0 :
	    print >> outFile, "Anomaly found in execution ", run, " in", anomalyCount, " samples"
	    anomalousRunCount += 1
    
  return anomalousRunCount

def runTrainedAutoencoder( model, testDataArray, datasetHeader, thresholdLoss, outFile ):
  
  #print(model.score(testDataArray))
  
  
  #encoded_data = model.encode(testDataArray) #TODO is there a difference between predict vs encode+decode
  #decoded_data = model.decode(encoded_data) 
  decoded_data = model.predict(testDataArray)
  dataLen, anomalyCount, ranking = detectAnomalyPoints(testDataArray, decoded_data, outFile, datasetHeader, thresholdLoss)

  ##debug print
  #print "----ranking-->" , ranking
  print >>  outFile, ranking

  ##debug print end

  
  
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

  #normalize with max : keeps the number between [0 to 1]
  
  #normalize min max mormalization
  minDataset = np.min(dataset, axis=0)
  maxDataset = np.max(dataset, axis=0)
  diff = dataset - minDataset
  minMaxDiff = maxDataset -  minDataset
  with np.errstate(divide='ignore', invalid='ignore'):
    c = np.true_divide(diff, minMaxDiff)
    c[c == np.inf] = 0
    c = np.nan_to_num(c)
  return c


def getTrainDataDir(dataDir, testDir=None, validationDir=None ):
  runs = os.listdir(dataDir)
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
  
 

def testModelAccuracy( model, outFilename, threshold_error=None ):
  
  perfTrainDataDir=sys.argv[1]
  perfTestDataDir=sys.argv[2] 
  perfValidDataDir = sys.argv[3] 

  if threshold_error == None:
    runs = os.listdir(perfTrainDataDir) 
    threshold_error =  getReconstructionErrorThreshold( model, perfTrainDataDir, runs )
    print "reconsturciton error: ", threshold_error

  outFile = open(outFilename, 'w')
  ## test for anomalous dataset
  runs = os.listdir(perfTestDataDir)

  
  anomalousRunCount = testAutoencoder( model, perfTestDataDir, runs, outFile, threshold_error ) 	
    
  truePositive = anomalousRunCount ## a
  falseNegative = len(runs) - anomalousRunCount ## b
  print >> outFile, "Actual Positive", len(runs)
  print >> outFile, "True Positive", anomalousRunCount
  print >> outFile, "False Negative", len(runs) - anomalousRunCount 
  
  print "Total run ", len(runs)
  print "Total anomalous run found ", anomalousRunCount
  

  ##validation with correct(not anomalous)  dataset	
  runs = os.listdir(perfValidDataDir)

  anomalousRunCount = testAutoencoder( model, perfValidDataDir, runs, outFile, threshold_error ) 	
  
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
  
  
  print "Report: ", outFilename


  

def perfAnalyzerMainTrainSequence(outputDir, networkConfig, training_sequence, threshold_final=None , saveTrainedNetwork=False):
  perfTrainDataDir=sys.argv[1]
  perfTestDataDir=sys.argv[2] 
  perfValidDataDir = sys.argv[3]
  #outputDir = sys.argv[4]
  outputFilePrefix = outputDir + "/" + "report."


  model = None
  logFileName = outputFilePrefix + "log"
  log_file = open(logFileName, 'w')
  
  #training_runs = os.listdir(perfTrainDataDir)
  #assert len(training_runs) >= configs.EXPERIMENT_EPOCHS[-1]

  #for epoch in configs.EXPERIMENT_EPOCHS:
  epoch_count = 0
  train_loss_list = []
  reconstruction_error_list = []
  #for i in range(len(training_sequence)): #TODO: tmp comment remove the next line
  for i in range(1): 
    train_run = training_sequence[i]
    epoch_count += 1
    print ("Training with", train_run)
    datadir = perfTrainDataDir + "/" + train_run
  
    datasetHeader, dataset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )


    #while len(dataset)  < networkConfig[0] * 2: ##TODO:tmp comment remove the next line
    for i in range(1, len(training_sequence)):
   
      print ("small input, adding more data for the batch")
      i += 1
      if(i == len(training_sequence)):
        break ## handle error
      print ("adding ", training_sequence[i])
      datadir = perfTrainDataDir + "/" + training_sequence[i] 
      redundantHeader, additionalDatatset = getPerfDataset( datadir , configs.NUMBER_OF_COUNTERS )
      dataset.extend(additionalDatatset)
  
    dataArray = getDatasetArray(dataset)
    trainingDataset = preprocessDataArray(dataArray)
    if len(dataset) < networkConfig[0] * 2:
      print "Not enough data for training this iteration" 
      break
    if model == None:
      model, train_loss  = trainAutoencoder( networkConfig, trainingDataset, trainingDataset, log_file )
      train_loss_list.append(train_loss)
    else:
      model, train_loss  = trainAutoencoder( networkConfig, trainingDataset, trainingDataset, log_file, model )
      train_loss_list.append(train_loss)
  if saveTrainedNetwork == True :
    model.save(outputDir + "/" + configs.MODEL_SAVED_FILE_NAME + "_"+ str(i))
  if threshold_final !=None:
    testModelAccuracy( model, outputDir+"/report.accuracy_"+ str(i), threshold_final )
  else:
    reconstruction_error_list.append(getReconstructionErrorThreshold( model, perfTrainDataDir, training_sequence )) 

  return model, train_loss_list, reconstruction_error_list

    

def findBestNetwork(inputLen, networkPrefix, networkSuffix, dataDir, outputDir):
  ##set network configs
  
  configs.setNetworkConfigs( networkPrefix, networkSuffix, inputLen )

  ## This is for training different network and choose a thresold
  ## adding more batch for more epochs
  result_runs  = getTrainDataDir(dataDir)
  #result_runs = []
  #for i in range(1):
	#  result_runs.extend(runs) 
  
  print ("Total execution found: ", len(result_runs))
  networkLogFile = open(outputDir + "/network_training.log", 'w')

  minimum_loss = None 
  minimum_loss_network = None
  best_model = None
  for key in configs.NETWORK_CONFIGS:
    print (key , configs.NETWORK_CONFIGS[key])
    print >> networkLogFile, key, configs.NETWORK_CONFIGS[key]
    experimentOutputDirName = outputDir + "/" + key
    mkdir_p(experimentOutputDirName)
    model, training_loss_list, reconstructionErrors = perfAnalyzerMainTrainSequence(experimentOutputDirName, configs.NETWORK_CONFIGS[key], result_runs)
    training_loss = training_loss_list[-1]
    print >> networkLogFile, training_loss_list, reconstructionErrors
    if minimum_loss == None or minimum_loss > training_loss:
      minimum_loss = training_loss
      minimum_loss_network = key
      best_model = model
  print "Best candidate: " ,minimum_loss_network,  configs.NETWORK_CONFIGS[minimum_loss_network], " with loss ", minimum_loss

  print >> networkLogFile, "Best candidate: " ,minimum_loss_network,  configs.NETWORK_CONFIGS[minimum_loss_network], " with loss ", minimum_loss

  networkLogFile.close()
  return minimum_loss_network, best_model, minimum_loss
  

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

def trainingTest(netConfig):
  testDir = None
  validationDir = None
  dataDir = sys.argv[1]
  testDir = None 
  if len(sys.argv) > 2 :
    testDir = sys.argv[2]
  
  if len(sys.argv) > 3 :
    validationDir = sys.argv[3]
 
  outputDir = sys.argv[4]

  result_runs  = getTrainDataDir(dataDir)
  mkdir_p(outputDir)
  model, training_loss_list, reconstructionErrors = perfAnalyzerMainTrainSequence(outputDir, netConfig, result_runs)
  testModelAccuracy( model, outputDir+"/report.accuracy" )
 
  

if __name__ == "__main__" :
  
  if(len(sys.argv) == 1):
    print ("Running Unit Test")
    unitTest()
    sys.exit()
   
  if len(sys.argv) > 5:
    if sys.argv[5] == "testrun":
      trainingTest(configs.NETWORK_CONFIGS['exp1'])
      sys.exit()
    else:
      print "Too many parameters"
      sys.exit()
  ##analyzing the dataset
  testDir = None
  validationDir = None
  dataDir = sys.argv[1]
  testDir = None 
  if len(sys.argv) > 2 :
    testDir = sys.argv[2]
  
  if len(sys.argv) > 3 :
    validationDir = sys.argv[3]

  
  outputDir = sys.argv[4]
  
  layerResults = [] ## tuple of network desc , error , trained_model

  oneLayerOutputDir = outputDir+'/onelayer'
  mkdir_p(oneLayerOutputDir)
 
  inputLen = configs.NUMBER_OF_COUNTERS
  networkPrefix = [ inputLen ]
  networkSuffix = [ ('tied', inputLen) ] 
  networkNameKey, trainedModel,reconstructionError =  findBestNetwork(inputLen, networkPrefix, networkSuffix, dataDir, oneLayerOutputDir)
  layerResults.append( (configs.NETWORK_CONFIGS[networkNameKey], reconstructionError, trainedModel) ) 
  #testModelAccuracy( trainedModel, oneLayerOutputDir+"/report.accuracy_1", reconstructionError )

  ## two layer experiments
  twoLayerOutputDir = outputDir+'/twolayers'
  mkdir_p(twoLayerOutputDir)

  ##Following Comment code does: for 2nd layer work on only the best of first layer    
  ## but we want to exaust all option for single layer too, thats what is done after the commented block 
  """  
  #singleLayerNetworkPerfix = configs.getNetworkPrefix(configs.NETWORK_CONFIGS[networkNameKey], 2)
  #singleLayerNetworkSuffix = configs.getNetworkSuffix(configs.NETWORK_CONFIGS[networkNameKey], 2)

  #encodedNodeNumber = configs.getEncodeLen(singleLayerNetworkPerfix)
      
  #networkNameKey, twoLayerTrainedModel,twoLayerReconstructionError =  findBestNetwork( encodedNodeNumber, singleLayerNetworkPerfix, singleLayerNetworkSuffix, dataDir, twoLayerOutputDir)
   
  #testModelAccuracy( trainedModel, twoLayerOutputDir+"/report.accuracy_2", reconstructionError )

  #print "single layer", configs.NETWORK_CONFIGS[networkNameKey], reconstructionError
  #print "two layer", configs.NETWORK_CONFIGS[networkNameKey], twoLayerReconstructionError
  """
  oneLayerEncodedNumbers = []
  oneLayerNetworkPerfix = []
  oneLayerNetworkSuffix = []
  for key in configs.NETWORK_CONFIGS:
    netPrefix = configs.getNetworkPrefix(configs.NETWORK_CONFIGS[key], 2)  
    netSuffix = configs.getNetworkSuffix(configs.NETWORK_CONFIGS[key], 2)  
    encodedNodeNumber = configs.getEncodeLen(netPrefix)
    oneLayerEncodedNumbers.append(encodedNodeNumber)
    oneLayerNetworkPerfix.append(netPrefix)
    oneLayerNetworkSuffix.append(netSuffix)
    

  numberOfOneLayerNetworks = len(oneLayerEncodedNumbers)
  minReconstructionError = None  
  bestNetworkNameKey = None
  bestModel = None
  for i in range(numberOfOneLayerNetworks):
    netPrefix = oneLayerNetworkPerfix[i]
    netSuffix = oneLayerNetworkSuffix[i]
    encNumber = configs.getEncodeLen(netPrefix)
    outputSubDir = twoLayerOutputDir + "/hidden1stLayerNode" + str(encNumber)
    mkdir_p(outputSubDir)
    networkNameKey, twoLayerTrainedModel,twoLayerReconstructionError =  findBestNetwork( encNumber, netPrefix, netSuffix, dataDir, outputSubDir)

    if bestModel == None or twoLayerReconstructionError < minReconstructionError:
      minReconstructionError = twoLayerReconstructionError
      bestNetworkNameKey = networkNameKey
      bestModel = twoLayerTrainedModel
        
  layerResults.append( (configs.NETWORK_CONFIGS[networkNameKey], minReconstructionError, bestModel))

  for resTuple in layerResults:
    print resTuple[0], resTuple[1]

 
  sortedByError = sorted(layerResults, key=lambda tup:tup[1])

   
  #testModelAccuracy( sortedByError[0][2], outputDir+"/report.accuracy", sortedByError[0][1] )#TODO: training error not matching reconstrucion error???
  testModelAccuracy( sortedByError[0][2], outputDir+"/report.accuracy" )
  print sortedByError[0]
  print sortedByError[1]
