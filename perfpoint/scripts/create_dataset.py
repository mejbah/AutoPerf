import sys
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)

PERFDATADIR=sys.argv[1]
NUM_OF_TOP_LEVEL_DIR=3
TOP_LEVEL_DIR_NAME=PERFDATADIR + "/outputs"
NUM_OF_RUN=12
INDEX_OF_TOT_INS=2

#MAX_NUM_OF_THREADS=32 
#MAX_MARK_ID=1


dataset = []
for run in range(NUM_OF_RUN):
  #collect all different perf counter from different outputs directory
  number_of_rows = 0
  number_of_cols = 0
  samples_from_one_run = []
  for i in range(1,NUM_OF_TOP_LEVEL_DIR+1):
	datadir= TOP_LEVEL_DIR_NAME+ str(i) + "/" + "test_" + str(run) + "_perf_data.csv"
	datarows = np.loadtxt(open(datadir,'rb'), delimiter=',', skiprows=3)
	number_of_rows = datarows.shape[0] #number of sample in this run
	#number_of_cols = datarows.shape[1]
 
	for row in range(number_of_rows):
	  current_row = datarows[row]
	  total_instruction = current_row[INDEX_OF_TOT_INS] 
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

#pp.pprint(dataset)
print len(dataset)
