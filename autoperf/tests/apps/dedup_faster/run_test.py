#!/usr/bin/python

import os
import sys
import subprocess
import re

# argv[1] : APP_NAME
# argv[2] : GOOD, BAD_MA, BAD_FS
# argv[3] : OUTPUT_DIR

if len(sys.argv) < 2:
  print "Usage: run_test.py OUTPUT_DIR\n"
  sys.exit() 

OUTPUT_DIR=sys.argv[1] 


DEFAULT_OUTFILE_NAME = "perf_data.csv"

EVENT_NUM = 39 #13




runCount = 0;

currOutputDirName = OUTPUT_DIR + '/run_' + str(runCount)
runCount += 1
p = subprocess.Popen(['mkdir', '-p', currOutputDirName]) #stdout=subprocess.PIPE, stderr=subprocess.PIPE)
p.wait()

TEST_ARGS = "native 16 threads"

##TODO:
os.environ["LD_LIBRARY_PATH"] = "/home/mejbah/WorkInProgress/perfpoint/source"
for i in range(1, EVENT_NUM):
  os.environ["PERFPOINT_EVENT_INDEX"] = str(i)
  start_time = os.times()[4]
  
  p = subprocess.Popen(['make', 'eval-perfpoint'])#stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  p.wait()
  time = os.times()[4] - start_time
  
  #copy data to a new file
  new_out_filename = currOutputDirName +"/event_" + str(i) + "_" + DEFAULT_OUTFILE_NAME
  out_file = open(new_out_filename, 'w')
  in_file = open(DEFAULT_OUTFILE_NAME, 'r')

  out_file.write('INPUT: ' + TEST_ARGS)
  out_file.write('\n')
  out_file.write('TIME: ' + str(time))
  out_file.write('\n')
  
  for line in in_file.readlines():
    out_file.write(line);

  in_file.close()
	#remove the original file
  p = subprocess.Popen( ['rm', DEFAULT_OUTFILE_NAME])
  p.wait()

  out_file.close()




