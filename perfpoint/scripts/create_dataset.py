import sys
import numpy as np
import pprint

import theanets
import numpy as np

THRESHOLD_ERROR=0.5


def createDataset(numberOfRun, topLevelDirName):
  dataset = []
  ##TODO: following two are hard coded values colsely related to the way perfpoint tool generates outputs and run_test.py stores output data
  indexOfTotalInstructionCol=2
  NUM_OF_TOP_LEVEL_DIR=3
  for run in range(numberOfRun):
    #collect all different perf counter from different outputs directory
    number_of_rows = 0
    number_of_cols = 0
    samples_from_one_run = []
    for i in range(1,NUM_OF_TOP_LEVEL_DIR+1):
	  datadir= topLevelDirName + str(i) + "/" + "test_" + str(run) + "_perf_data.csv"
	  datarows = np.loadtxt(open(datadir,'rb'), delimiter=',', skiprows=3)
	  number_of_rows = datarows.shape[0] #number of sample in this run
	  #number_of_cols = datarows.shape[1]
   
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
  return dataset 



def findMispredicts(realData, predictedData):
  #for x  in realData:
  outFile = open("root.out",'w') #TODO: output file as param rather than local
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
  outFile.close() 



def perfAutoencoder(trainDataPath, numberOfTrainRun, testDataPath=None, numberOfTestRun=0):
  
  topLevelDirName = trainDataPath + "/outputs"
  trainingData = createDataset(numberOfTrainRun, topLevelDirName)

    
  #make np array
  trainingDataArray = np.array(trainingData, dtype='float32')
  
  
  ############################
  # Autoencoder using theanet#
  ############################
  
  
  # 1. create a model 
  inputVectorLen = 14
  assert inputVectorLen == trainingDataArray.shape[1]
  outputVectorLen = 14
  hiddenLayer = 20
  
  model = theanets.Autoencoder([inputVectorLen, hiddenLayer, outputVectorLen]) #data array with 14 columns
  
  
  # optional: set up additional losses.
  #model.add_loss('mae', weight=0.1)
  
  
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
	testData =  createDataset(numberOfTestRun, topLevelDirName)
	testDataArray = np.array(testData)
	#print(model.predict(test_data))
	print(model.score(testDataArray))
	encoded_data = model.encode(testDataArray)
	print encoded_data.shape
	decoded_data = model.decode(encoded_data)

	findMispredicts(testDataArray, decoded_data)	
  
  
  
  #if perform_test == True:
  


if __name__ == "__main__" :
  
  #pp = pprint.PrettyPrinter(indent=4)

  perfTrainDataDir=sys.argv[1]
  #numberOfTrainRun = 12 #TODO: maximum 12 run we can use but split them for validation
  numberOfTrainRun = 2
  #MAX_NUM_OF_THREADS=32 
  #MAX_MARK_ID=1

  ##if we want to test some data
  if len(sys.argv) == 3:
	perfTestDataDir=sys.argv[2]
	numberOfTestRun = 2
	perfAutoencoder(perfTrainDataDir, numberOfTrainRun, perfTestDataDir, numberOfTestRun)

  else:
	perfAutoencoder(perfTrainDataDir, numberOfTrainRun)
	




