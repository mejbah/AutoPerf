import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg') #instead of Xserver for png
import matplotlib.pyplot as plt

import pprint

import theanets
import numpy as np
import dA 
import plot_utils



THRESHOLD_ERROR=0.5


def plotPerfCounters( datasetArray1, datasetArray2, column, columnName, outputDir ):
  
  fig = plt.figure()
  ax = fig.add_subplot(111)

  x_values = np.arange(datasetArray1.shape[0])
  y_values = datasetArray1[:,column] 
  
  ax.plot(x_values, y_values, 'ro')

  x_values = np.arange(datasetArray2.shape[0])
  y_values = datasetArray2[:,column] 
  
  ax.plot(x_values, y_values, 'bo')
  ax.set_ylabel(columnName)

  fig.savefig( outputDir + '/counter_'+ columnName + '.png')


def createDataset(numberOfRun, topLevelDirName):
  dataset = []
  ##TODO: following two are hard coded values colsely related to the way perfpoint tool generates outputs and run_test.py stores output data
  indexOfTotalInstructionCol=2
  NUM_OF_TOP_LEVEL_DIR=3

  #save the header information for clarity of constructed dataset
  datasetHeader = []
  for i in range(1,NUM_OF_TOP_LEVEL_DIR+1):
	datadir= topLevelDirName + str(i) + "/" + "test_0_perf_data.csv"  #read only first run for getting headers, others should be same
	with open(datadir,'r') as fp:
	  for linenumber,line in enumerate(fp):
		if linenumber == 2: #3rd line is the header
		  headers = line.strip().split(',')
		  if i==1:
			datasetHeader = headers[0:2]
		  datasetHeader.extend(headers[3:])
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
		total_instruction = current_row[indexOfTotalInstructionCol] 
		current_data = current_row[3:]
		#TODO: are these first two unnormalized values are important or creating problem?
		if i==1: 
		  samples_from_one_run.append(current_row[0:2].tolist()) # first add the first two column with id and #of threads
		normalized_data = current_data /total_instruction	
		samples_from_one_run[row].extend(normalized_data.tolist()) 
  
    if run == 0 :
	 dataset = samples_from_one_run
    else:
	 dataset.extend(samples_from_one_run)
  return dataset, datasetHeader 



def findMispredicts(realData, predictedData, outFile):
  #for x  in realData:
  datasetLen = realData.shape[0]
  dataLen = realData.shape[1]
  for x  in range(datasetLen):
	anomaly_found = False
	for y in range(2, dataLen):
	  #if realData[x][y] != predictedData[x][y]:
	  if (abs(realData[x][y])-abs(predictedData[x][y])) / abs(realData[x][y]) > THRESHOLD_ERROR: #TODO: taking the abs as Autoencoder predicts negative values,FIX IT
		if anomaly_found == False:
		  outFile.write(str(realData[x][0]))
		  anomaly_found = True
		outFile.write(" " + str(y))
	if anomaly_found == True:
	  outFile.write("\n")



def plotDataset(trainDataPath, numberOfTrainRun, testDataPath, numberOfTestRun, outputDir):
  #################
  ##plot dataset###
  #################
  topLevelDirName = trainDataPath + "/outputs"
  trainingData, datasetHeader = createDataset(numberOfTrainRun, topLevelDirName)
  #make np array
  trainingDataArray = np.array(trainingData, dtype='float32')

  topLevelDirName = testDataPath + "/outputs"
  assert numberOfTestRun > 0
  testData, datasetHeader=  createDataset(numberOfTestRun, topLevelDirName)
  testDataArray = np.array(testData)
  column = 3 #TODO: change this according to need
  print "ploting data for ", datasetHeader[column]
  print trainingDataArray.shape[0]
  print testDataArray.shape[0]
  plotPerfCounters(trainingDataArray, testDataArray, column, datasetHeader[column], outputDir)



def perfAutoencoder(trainDataPath, numberOfTrainRun, testDataPath=None, numberOfTestRun=0, outputFile=None):
  
  topLevelDirName = trainDataPath + "/outputs"
  trainingData, datasetHeader = createDataset(numberOfTrainRun, topLevelDirName)

    
  #make np array
  trainingDataArray = np.array(trainingData, dtype='float32')
 
  
  ############################
  # Autoencoder using theanet#
  ############################
  
  
  # 1. create a model 
  inputVectorLen = 14
  assert inputVectorLen == trainingDataArray.shape[1]
  outputVectorLen = 14
  hiddenLayer = 10
  
  model = theanets.Autoencoder([inputVectorLen, hiddenLayer, outputVectorLen]) #data array with 14 columns
  #model = theanets.Autoencoder([inputVectorLen, hiddenLayer, (outputVectorLen, 'tied')]) #data array with 14 columns
  
  
  # optional: set up additional losses.
  #model.add_loss('mae', weight=0.1)
  
  print "Training autoencoder with dateset " , trainingDataArray.shape[0]
  # 2. train the model.
  #In one method, zero-mean Gaussian noise is added to the input data or hidden representations. 
  #These are specified during training using the input_noise and hidden_noise keyword arguments, respectively.The value of the argument specifies the standard deviation of the noise.
  model.train([trainingDataArray], input_noise=0.1, hidden_noise=0.1)
  #model.train(
  #    training_data,
  #    validation_data,
  #    algo='rmsprop',
  #    hidden_l1=0.01,  # apply a regularizer.
  #)
  
 
   
  # 3. use the trained model.
  if testDataPath != None:
	topLevelDirName = testDataPath + "/outputs"
	assert numberOfTestRun > 0
	testData, datasetHeader=  createDataset(numberOfTestRun, topLevelDirName)
	testDataArray = np.array(testData)
	
	print "Test autoencoder with dateset " , testDataArray.shape[0]
	#print(model.predict(test_data))
	print(model.score(testDataArray))
	encoded_data = model.encode(testDataArray)
	print encoded_data.shape
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
	outFile.close() 
 
	

	 
  #if perform_test == True:
  


 
def theanetAutoencoder():
  #pp = pprint.PrettyPrinter(indent=4)
  perfTrainDataDir=sys.argv[1]
  #numberOfTrainRun = 12 #TODO: maximum 12 run we can use but split them for validation 
  numberOfTrainRun = 8
  #MAX_NUM_OF_THREADS=32 
  #MAX_MARK_ID=1
  
  
  ##if we want to test some data
  if len(sys.argv) > 2:
	perfTestDataDir=sys.argv[2]
	numberOfTestRun = 2
	perfAutoencoder(perfTrainDataDir, numberOfTrainRun, perfTestDataDir, numberOfTestRun)

  else:
	perfAutoencoder(perfTrainDataDir, numberOfTrainRun)
  
  
 
def customAutoencoder():

  perfTrainDataDir=sys.argv[1]
  #numberOfTrainRun = 12 #TODO: maximum 12 run we can use but split them for validation
  numberOfTrainRun = 2
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






if __name__ == "__main__" :
  
  #pp = pprint.PrettyPrinter(indent=4)
  ##############################################
  ## plot data
  #############################################  
  perfTrainDataDir=sys.argv[1]
  numberOfTrainRun = 8
  perfTestDataDir=sys.argv[2]
  numberOfTestRun = 2
  outputDir = sys.argv[3]
  plotDataset(perfTrainDataDir, numberOfTrainRun, perfTestDataDir, numberOfTestRun, outputDir)


  ################################################
  #RUN TRAIN and TEST #
  #####################################
  #customAutoencoder()
  theanetAutoencoder() ##use this one
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
	




