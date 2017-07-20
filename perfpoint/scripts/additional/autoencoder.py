import sys
import climate
import numpy as np
import matplotlib as mpl
mpl.use('Agg') #instead of Xserver for png
import matplotlib.pyplot as plt

import pprint

import theanets
import numpy as np
import dA 
import plot_utils
import os
import errno

## log results ###
climate.enable_default_logging()

THRESHOLD_ERROR=0.5


def mkdir_p(path):
  try:
	os.makedirs(path)
  except OSError as exc:  # Python >2.5
	if exc.errno == errno.EEXIST and os.path.isdir(path):
	  pass
	else:
	  raise

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
returns number of executions profile csv file present in the directory
"""
def getNumberOfExecProfile( dirName ):
  profiles = [ file.endswith(".csv") for file in dirName]
  return len(profiles)

def getExecProfileFileNames( dirName ):
  profiles = [ file.endswith(".csv") for file in dirName]
  return len(profiles)





def createDataset(numberOfRun, topLevelDirName):
  dataset = []
  ##TODO: following two are hard coded values colsely related to the way perfpoint tool generates outputs and run_test.py stores output data
  indexOfTotalInstructionCol=2
  indexOfTotalThreads=1
  NUM_OF_TOP_LEVEL_DIR=3
  #NUM_OF_TOP_LEVEL_DIR=1

  #save the header information for clarity of constructed dataset
  datasetHeader = []
  for i in range(1,NUM_OF_TOP_LEVEL_DIR+1):
	datadir= topLevelDirName + str(i) + "/" + "test_0_perf_data.csv"  #read only first run for getting headers, others should be same
	with open(datadir,'r') as fp:
	  for linenumber,line in enumerate(fp):
		if linenumber == 2: #3rd line is the header
		  headers = line.strip().split(',')
		  if i==1:
		  	datasetHeader = headers[1:2]
		  datasetHeader.extend(headers[3:])
		  #if i==1:
		  #	datasetHeader = headers[3:]
		  #else:
		  #	datasetHeader.extend(headers[3:])
		  break

  
  #collect the real data excluding the header
  for run in range(numberOfRun):
    #collect all different perf counter from different outputs directory
    number_of_rows = 0
    number_of_cols = 0
    samples_from_one_run = []
    for i in range(1,NUM_OF_TOP_LEVEL_DIR+1):
	  datadir= topLevelDirName + str(i) + "/" + "test_" + str(run) + "_perf_data.csv"
	  datarows = np.loadtxt(open(datadir,'rb'), delimiter=',', skiprows=3) # first two rows contains information of the particular execution and 3rd row is the header
	  number_of_rows = datarows.shape[0] #number of sample in this run
	  #number_of_cols = datarows.shape[1]
	
	  # now first(0th)  line is the header
	  for row in range(number_of_rows): 
		current_row = datarows[row]
		total_instruction = current_row[indexOfTotalInstructionCol] #for normalization
		total_threads = current_row[indexOfTotalThreads] #for normalization
		current_data = current_row[3:]
		#normalized_data = current_data / total_instruction
		normalized_data = current_data / (total_instruction	* total_threads)
		#normalized_data = current_data #skip normalization as we are doing preprocessing
		#TODO: are these first two unnormalized values are important or creating problem?
		if i==1: 
		  samples_from_one_run.append(current_row[1:2].tolist()) # first add the first two column with annotated unique id and #of threads	
		
		samples_from_one_run[row].extend(normalized_data.tolist()) 
		#if i==1:
		#  samples_from_one_run.append(normalized_data.tolist()) 
		#else:
		#  samples_from_one_run[row].extend(normalized_data.tolist()) 
  
    if run == 0 :
	 dataset = samples_from_one_run
    else:
	 dataset.extend(samples_from_one_run)
  return dataset, datasetHeader 


def getNormalizedDistance( old, new ):
  dist = np.linalg.norm(old-new)
  origin = np.linalg.norm(old)
  return dist/origin



"""
calculate reconstruction error : normalized distance 
"""
def detectAnomalyPoints( realData, predictedData, outFile ):
  #for x  in realData:
  datasetLen = realData.shape[0]
  dataLen = realData.shape[1]
  for x  in range(datasetLen):
	reconstructionError = getNormalizedDistance( realData[x], predictedData[x] )
	if(reconstructionError > THRESHOLD_ERROR):
	  outputStr = "No." + str(x) + " "
	  for y in range(0, dataLen):
		outputStr += str(y)
	  	outputStr += ":"
	  	outputStr += str(abs(predictedData[x][y]-realData[x][y]))
		outputStr += " "
	  outputStr += str(reconstructionError)
	  outFile.write(outputStr)
	  outFile.write("\n")



def findMispredicts(realData, predictedData, outFile):
  #for x  in realData:
  datasetLen = realData.shape[0]
  dataLen = realData.shape[1]
  for x  in range(datasetLen):
	anomaly_found = False
	for y in range(0, dataLen):
	  #if realData[x][y] != predictedData[x][y]:
	  #if (abs(realData[x][y])-abs(predictedData[x][y])) / abs(realData[x][y]) > THRESHOLD_ERROR: #TODO: taking the abs as Autoencoder predicts negative values,FIX IT
	  predictionError = (realData[x][y]-predictedData[x][y]) / realData[x][y]
	  if  predictionError > THRESHOLD_ERROR: #TODO: taking the abs as Autoencoder predicts negative values,FIX IT
		if anomaly_found == False:
		  anomaly_found = True
		outFile.write(" " + str(y) + ":" + str(predictionError) )
	if anomaly_found == True:
	  outFile.write("\n")



def  plotModelOutput( testDataPath, numberOfTestRun, decoded_dataset, outputDir):
  

  topLevelDirName = testDataPath + "/outputs"
  assert numberOfTestRun > 0
  testData, datasetHeader=  createDataset(numberOfTestRun, topLevelDirName)
  testDataArray = np.array(testData)
  
  total_perf_counter = testDataArray.shape[1]
  assert testDataArray.shape[0] == decoded_dataset.shape[0]

  for column in range(0, total_perf_counter):
	print "ploting data for ", datasetHeader[column]
	plotPerfCounters( testDataArray, decoded_dataset, column, datasetHeader[column], outputDir)
 

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

def trainAutoencoder( trainingDataArray, hiddenLayerLen ):
  inputVectorLen = trainingDataArray.shape[1]
  outputVectorLen = trainingDataArray.shape[1]
  
  #model = theanets.Autoencoder([inputVectorLen, hiddenLayerLen, outputVectorLen]) #data array with 14 columns
  #model = theanets.Autoencoder([inputVectorLen, (hiddenLayerLen,'relu'), ('tied', outputVectorLen )]) #data array with 14 columns
  #model = theanets.Autoencoder(layers=(inputVectorLen, (hiddenLayerLen,'relu'), (outputVectorLen, 'tied', 'relu' ))) #data array with 14 columns
  ## tied weights has to have palindromic network
  model = theanets.Autoencoder([inputVectorLen, (8,'relu'), ( 6,'relu'), ('tied', 8,'relu'), ('tied',outputVectorLen)]) #data array with 14 columns
 
  # optional: set up additional losses.
  #model.add_loss('mae', weight=0.1)
  
  print "Training autoencoder with dateset " , trainingDataArray.shape[0]
  # 2. train the model.
  #In one method, zero-mean Gaussian noise is added to the input data or hidden representations. 
  #These are specified during training using the input_noise and hidden_noise keyword arguments, respectively.
  #The value of the argument specifies the standard deviation of the noise.

  #model.train([trainingDataArray], input_noise=0.1, hidden_noise=0.01)
  #model.train([trainingDataArray])

  model.train(trainingDataArray, 
          #algo='layerwise',
          #train_batches=10, #trainingDataArray.shape[0],
          #learning_rate=0.01,
          #momentum=0.9,
          #hidden_dropout=0.5)
		  input_noise=0.1)


  score = model.score(trainingDataArray)
  #model.train([trainingDataArray], hidden_l1=0.1 ) #sparsity penalty
  #model.train(
  #    training_data,
  #    validation_data,
  #    algo='rmsprop',
  #    hidden_l1=0.01,  # apply a regularizer.
  #)
  return model, score



def runTrainedAutoencoder( model, testDataArray, datasetHeader, outputFile=None ):
  
  print "Test score " 
  print(model.score(testDataArray))
  
  
  encoded_data = model.encode(testDataArray)
  decoded_data = model.decode(encoded_data)
  if outputFile == None:
    outFile = open("root.out",'w') #TODO: output file as param rather than local
  else:
	outFile = outputFile + "/report.out"
	outFile = open(outFile,'w') #TODO: output file as param rather than local
  idx = 0
  for title in datasetHeader:
    outFile.write(str(idx) + "." + title + " || ")
    idx+=1
  outFile.write("\n")
  #findMispredicts(testDataArray, decoded_data, outFile)	
  detectAnomalyPoints(testDataArray, decoded_data, outFile)
  outFile.close() 
  
  return decoded_data


def createDataArray( datapath, numberofexec ):

  topLevelDirName = datapath + "/outputs"
  
  print "reading dataset\n"    
  dataset, datasetHeader = createDataset(numberofexec, topLevelDirName)

  print "dataset created\n"    
  #make np array
  dataArray = np.array(dataset, dtype='float32')

  return dataArray , datasetHeader
 
  

def perfAutoencoder(trainDataPath, numberOfTrainRun, testDataPath=None, numberOfTestRun=0, outputFile=None):
  
  topLevelDirName = trainDataPath + "/outputs"
  trainingData, datasetHeader = createDataset(numberOfTrainRun, topLevelDirName)

    
  #make np array
  trainingDataArray = np.array(trainingData, dtype='float32')
 
  
  ############################
  # Autoencoder using theanet#
  ############################
  
  
  # 1. create a model 
  #inputVectorLen = 12
  inputVectorLen = trainingDataArray.shape[1]
  assert inputVectorLen == trainingDataArray.shape[1]
  #outputVectorLen = 12
  outputVectorLen = trainingDataArray.shape[1]
  hiddenLayer =	8 #trainingDataArray.shape[1] / 2
  
  #model = theanets.Autoencoder([inputVectorLen, hiddenLayer, outputVectorLen]) #data array with 14 columns
  model = theanets.Autoencoder([inputVectorLen, (hiddenLayer,'sigmoid'), (outputVectorLen, 'tied')]) #data array with 14 columns
  
  
  # optional: set up additional losses.
  #model.add_loss('mae', weight=0.1)
  
  print "Training autoencoder with dateset " , trainingDataArray.shape[0]
  # 2. train the model.
  #In one method, zero-mean Gaussian noise is added to the input data or hidden representations. 
  #These are specified during training using the input_noise and hidden_noise keyword arguments, respectively.The value of the argument specifies the standard deviation of the noise.
  model.train([trainingDataArray], input_noise=0.1, hidden_noise=0.1)
  #model.train([trainingDataArray])
  print "Training score " 
  print(model.score(trainingDataArray))
 # model.train([trainingDataArray], hidden_l1=0.1 ) #sparsity penalty
  #model.train(
  #    training_data,
  #    validation_data,
  #    algo='rmsprop',
  #    hidden_l1=0.01,  # apply a regularizer.
  #)
  
 
  decoded_data = None 
  # 3. use the trained model.
  if testDataPath != None:
	topLevelDirName = testDataPath + "/outputs"
	assert numberOfTestRun > 0
	testData, datasetHeader=  createDataset(numberOfTestRun, topLevelDirName)
	testDataArray = np.array(testData)
	
	print "Test autoencoder with dateset " , testDataArray.shape[0]
	#print(model.predict(test_data))
	
	print "Test score " 
	print(model.score(testDataArray))

	encoded_data = model.encode(testDataArray)
	decoded_data = model.decode(encoded_data)
	if outputFile == None:
	  outFile = open("root.out",'w') #TODO: output file as param rather than local
	else:
	  outFile = open(outputFile,'w') #TODO: output file as param rather than local
	idx = 0
	for title in datasetHeader:
	  outFile.write(str(idx) + "." + title + " || ")
	  idx+=1
	outFile.write("\n")
	findMispredicts(testDataArray, decoded_data, outFile)	
	ouitFile.close() 
 
	
  return decoded_data
	 
  #if perform_test == True:
  


 
def theanetAutoencoder():
  #pp = pprint.PrettyPrinter(indent=4)
  perfTrainDataDir=sys.argv[1]
  #numberOfTrainRun = 12 #TODO: maximum 12 run we can use but split them for validation 
  numberOfTrainRun = 6
  #MAX_NUM_OF_THREADS=32 
  #MAX_MARK_ID=1
  
  
  ##if we want to test some data
  if len(sys.argv) > 2:
	perfTestDataDir=sys.argv[2]
	numberOfTestRun = 2
	outputFile = None
	if len(sys.argv) > 3:
	  outputDir=sys.argv[3]
	  outputFile=outputDir + "/root.out"
	perfAutoencoder(perfTrainDataDir, numberOfTrainRun, perfTestDataDir, numberOfTestRun, outputFile)

  else:
	perfAutoencoder(perfTrainDataDir, numberOfTrainRun)
  
  
 
def customAutoencoder():

  perfTrainDataDir=sys.argv[1]
  #numberOfTrainRun = 12 #TODO: maximum 12 run we can use but split them for validation
  numberOfTrainRun = 6
  #MAX_NUM_OF_THREADS=32 
  #MAX_MARK_ID=1
  topLevelDirName = perfTrainDataDir + "/outputs"
  trainingData, datasetHeader = createDataset(numberOfTrainRun, topLevelDirName)
  
  #make np array
  trainingDataArray = np.array(trainingData, dtype='float32')

  #choose specific columns
  #trainingDataset = trainingDataArray[:, [0,3,4,5,13]]
  trainingDataset = trainingDataArray[:, [0,3,4,5]]

  training_epochs = 10
  learning_rate = 0.1
  batch_size = trainingDataset.shape[0]
  n_input = trainingDataset.shape[1] 
  encode_size = 3
  #plotPerfCounters(trainingDataset,3)
  autoencoderInstance = dA.getInstance( n_input, encode_size)
  dA.trainAutoencoder( autoencoderInstance,trainingDataset, encode_size, training_epochs, learning_rate, batch_size )

  
  ##if we want to test some data
  if len(sys.argv) == 3:
	perfTestDataDir=sys.argv[2]
	numberOfTestRun = 2
	
	testData, datasetHeader = createDataset(numberOfTrainRun, topLevelDirName)

	testDataArray = np.array(testData, dtype='float32')

	#choose specific columns
	testDataset = trainingDataArray[:, [0,3,4,5]]
	batch_size = testDataset.shape[0]
	
	dA.testAutoencoder( autoencoderInstance,testDataset, batch_size )




def preprocessDataArray( dataset ):
  #zero centering
  dataset -= np.mean(dataset, axis=0)
  #normalize
  #dataset /= np.std(dataset, axis=0)

  return dataset


def perfAnalyzerMain():
  perfTrainDataDir=sys.argv[1]
  numberOfTrainRun = int(sys.argv[2])
  perfTestDataDir=sys.argv[3]
  numberOfTestRun = int(sys.argv[4])
  outputDir = sys.argv[5]

  trainingDataArray, trainDatasetHeader = createDataArray( perfTrainDataDir, numberOfTrainRun )
  
  testDataArray, testDatasetHeader  = createDataArray( perfTestDataDir, numberOfTestRun )

  #trainingDataset = trainingDataArray[:, [0,1,2,3,4,7,10,12]]
  #trainingHeaderset = [ trainDatasetHeader[i] for i in [0,1,2,3,4,7,10,12] ]
  #testDataset = testDataArray[:, [0,1,2,3,4,7,10,12]]
  #testHeaderset = [testDatasetHeader[i] for i in [0,1,2,3,4,7,10,12] ]
  
  trainingDataset = trainingDataArray[:, [0,1,2,3,4,7, 9, 10, 11, 12]]
  trainingHeaderset = [ trainDatasetHeader[i] for i in [0,1,2,3,4,7, 9, 10, 11, 12] ]
  testDataset = testDataArray[:, [0,1,2,3,4,7, 9, 10, 11, 12]]
  testHeaderset = [testDatasetHeader[i] for i in [0,1,2,3,4,7, 9, 10, 11, 12] ]

  datasetPlotPath = outputDir+"/dataset"
  mkdir_p(datasetPlotPath)
  plotDataset( trainingDataArray, testDataArray, testDatasetHeader, outputDir+"/dataset" )
  #plotDataset( trainingDataset, testDataset, testHeaderset, outputDir )

  #print "Training dataset ", trainingDataArray.shape
  print "Training dataset ", trainingDataset.shape
  #print "Test dataset " , testDataArray.shape
  print "Test dataset " , testDataset.shape
  
  trainingDataset = preprocessDataArray(trainingDataset)
  testDataset = preprocessDataArray(testDataset)
  
  model_selected = None
  max_score = 0
  for hidden in range(6,7):#range(1, trainingDataArray.shape[1]-1):
	#model, score  = trainAutoencoder( trainingDataArray, hidden )
	model, score  = trainAutoencoder( trainingDataset, hidden )
	print "hidden unit ", hidden, " score ", score
	if model_selected == None:
	  model_selected = model
	  max_score = score
	else:
	  if score > max_score:
		model_selected = model
		max_score = score

  
  print "Training max score ", max_score	
   
  #reconstructedData = runTrainedAutoencoder( model_selected, testDataArray, testDatasetHeader, outputDir )
  reconstructedData = runTrainedAutoencoder( model_selected, testDataset, testHeaderset, outputDir )

    
  #plotDataset( trainingDataArray, testDataArray, testDatasetHeader, outputDir ,reconstructedData )
  plotDataset( trainingDataset, testDataset, testHeaderset, outputDir ,reconstructedData )




if __name__ == "__main__" :
  
  
  
"""
  perfAnalyzerMain()
"""
  #pp = pprint.PrettyPrinter(indent=4)
 
"""


  ################################################
  #RUN TRAIN and TEST #
  #####################################
  #customAutoencoder()
  #theanetAutoencoder() ##use this one
  outputFile=outputDir + "/root.out"
  decoded_dataset = perfAutoencoder(perfTrainDataDir, numberOfTrainRun, perfTestDataDir, numberOfTestRun, outputFile)

  resultDir = sys.argv[4] 
  plotModelOutput(perfTestDataDir, numberOfTestRun, decoded_dataset, resultDir)

"""


""""
  perfTrainDataDir=sys.argv[1]
  #numberOfTrainRun = 12 #TODO: maximum 12 run we can use but split them for validation
  numberOfTrainRun = 2
  #MAX_NUM_OF_THREADS=32 
  #MAX_MARK_ID=1
  topLevelDirName = perfTrainDataDir + "/outputs"
  trainingData, datasetHeader = createDataset(numberOfTrainRun, topLevelDirName)
  
  #make np array
  print datasetHeader
  trainingDataArray = np.array(trainingData, dtype='float32')

  #choose specific columns
  #trainingDataset = trainingDataArray[:, [0,3,4,5,13]]
  trainingDataset = trainingDataArray[:, [0,3,4,5]]
  
  ##if we want to test some data
  if len(sys.argv) == 3:
	perfTestDataDir=sys.argv[2]
	numberOfTestRun = 2
	perfAutoencoder(perfTrainDataDir, numberOfTrainRun, perfTestDataDir, numberOfTestRun)

  else:
	#perfAutoencoder(perfTrainDataDir, numberOfTrainRun)
	training_epochs = 10
	learning_rate = 0.1
	batch_size = trainingDataset.shape[0]
	n_input = trainingDataset.shape[1] 
	encode_size = 3
	print trainingDataset.shape
	print trainingDataset
	#plotPerfCounters(trainingDataset,3)
	autoencoderInstance = dA.getInstance( n_input, encode_size)
	dA.trainAutoencoder( autoencoderInstance,trainingDataset, encode_size, training_epochs, learning_rate, batch_size )
	dA.testAutoencoder( autoencoderInstance,trainingDataset, batch_size )
"""
	




