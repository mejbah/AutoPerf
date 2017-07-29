#!/usr/bin/python

import os
import sys
import subprocess
import re



TEST_ARGS_FILE_NAME="TEST_ARGS"
APP_NAME="reverse_index"
DATASET_HOME="/home/mejbah/lockperf/multithreadingtests/datasets"
DEFAULT_OUTFILE_NAME="perf_data.csv"
OUTPUT_DIR="./outputs" + sys.argv[1] #argv[1] decides which set of event in 3 different eventset is running

#mkdir for output files
p = subprocess.Popen(['mkdir', '-p', OUTPUT_DIR]) 
p.wait()


f = open(TEST_ARGS_FILE_NAME, 'r')
nthreadList = []
numberOfTest = 0
for line in f.readlines():
  line = line.strip()
  if len(line)>0:
	nthreadList.append(line.split()[0])
	numberOfTest += 1
	


for i in range(numberOfTest):
  TEST_ARGS = DATASET_HOME+"/"+ APP_NAME +" "+nthreadList[i]

  start_time = os.times()[4]
  p = subprocess.Popen(['make', 'eval-perfpoint', 'TEST_ARGS='+TEST_ARGS]) #stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  p.wait()
  time = os.times()[4] - start_time

  #copy data to a new file
  new_out_filename = OUTPUT_DIR +"/test_" + str(i) + "_" + DEFAULT_OUTFILE_NAME
  out_file = open(new_out_filename, 'w')
  in_file = open(DEFAULT_OUTFILE_NAME, 'r')

  out_file.write(TEST_ARGS)
  out_file.write('\n')
  out_file.write(str(time))
  out_file.write('\n')

  for line in in_file.readlines():
	out_file.write(line);

  in_file.close()
  
  #remove the original file
  p = subprocess.Popen( ['rm', DEFAULT_OUTFILE_NAME])
  p.wait()

  out_file.close()
  
  
