#!/usr/bin/python

import os
import sys
import subprocess
import re

# argv[1] : APP_NAME
# argv[2] : GOOD, BAD_MA, BAD_FS
# argv[3] : OUTPUT_DIR

if len(sys.argv) < 3:
  print "Usage: run_test.py APP_NAME GOOD|BAD_FS OUTPUT_DIR\n"
  sys.exit() 

APP_NAME= sys.argv[1]
TYPE_FLAG = ""
if sys.argv[2]  == "GOOD":
  TYPE_FLAG ='-DNO_FALSE_SHARING'
OUTPUT_DIR=sys.argv[3] 


INPUT_DIR="/home/mejbah/datasets/reverse_index/"
DEFAULT_OUTFILE_NAME = "perf_data.csv"
EVENT_NUM = 40
NThread = [8,12]
#INFILE = ["large.bmp", "med.bmp"]
INFILE = ["dummy"]

#EVENT_NUM = 2
#NThread = [2] #3,4,5,6,7,8,9,10,11,12]
#NArray = [50000000] # 2000000, 5000000, 10000000, 20000000, 50000000, 100000000]



#mkdir for output files
#p = subprocess.Popen(['mkdir', '-p', OUTPUT_DIR]) 
#p.wait()

runCount = 0;

for n in INFILE:
  ##make CFLAGS="-DGOOD -DN=100000000"
  #REPEAT=10 #REPEAT FACTOR
  #MAKE_FLAGS = 'CFLAGS=' + TYPE_FLAG + ' -DREPEAT=' + str(REPEAT)   
  #print "-----",  MAKE_FLAGS
  #p = subprocess.Popen(['make', 'clean'])
  #p.wait()
  #p = subprocess.Popen(['make', MAKE_FLAGS])
  #p.wait()
  for thread in NThread:
    currOutputDirName = OUTPUT_DIR + '/run_' + str(runCount)
    runCount += 1
    p = subprocess.Popen(['mkdir', '-p', currOutputDirName]) #stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
 
    TEST_ARGS = INPUT_DIR + " " + str(thread)
    
    for i in range(EVENT_NUM):
      os.environ["PERFPOINT_EVENT_INDEX"] = str(i)
      start_time = os.times()[4]
      p = subprocess.Popen(['make', 'eval-perfpoint', 'TEST_ARGS='+TEST_ARGS]) #stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      p.wait()
      time = os.times()[4] - start_time
      
      #copy data to a new file
      new_out_filename = currOutputDirName +"/event_" + str(i) + "_" + DEFAULT_OUTFILE_NAME
      out_file = open(new_out_filename, 'w')
      in_file = open(DEFAULT_OUTFILE_NAME, 'r')
  
      out_file.write('INPUT: ' + str(n) + ' ' + str(thread))
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

  


