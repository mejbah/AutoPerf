#!/usr/bin/python

import os
import sys
import subprocess
import re

PERFPOINT_LIB_PATH="/home/mejbah/git_repos/Perf-Anomaly/profiler"
COUNTERS_FILE="COUNTERS"
PROG_BIN = ""
PROG_ARGS = ""
# argv[1] : OUTPUT_DIR

def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      if len(l) == 0:
        print "Empty line in ", fname
        sys.exit()
      pass
    return i + 1



if len(sys.argv) < 3:
  print "\nUsage: run_test.py OUTPUT_DIR PROG_BIN PROG_ARGS [run id]\n"
  sys.exit() 


OUTPUT_DIR=sys.argv[1] 
PROG_BIN=sys.argv[2]
runId = "run0"
if len(PROG_BIN)==0:
  print "Error:binary name required"
  sys.exit()

if len(sys.argv) == 4:
  PROG_ARGS = sys.argv[3]
if len(sys.argv) == 5:
  runId = sys.argv[4]

DEFAULT_OUTFILE_NAME = "perf_data.csv"
N_EVENTS = file_len(COUNTERS_FILE)

print "Profiling ", N_EVENTS, " events"

currOutputDirName = OUTPUT_DIR + '/' + runId
p = subprocess.Popen(['mkdir', '-p', currOutputDirName]) #stdout=subprocess.PIPE, stderr=subprocess.PIPE)
p.wait()


os.environ["LD_LIBRARY_PATH"] = PERFPOINT_LIB_PATH

print PROG_BIN, PROG_ARGS


for i in range(0,N_EVENTS):
  os.environ["PERFPOINT_EVENT_INDEX"] = str(i)
  start_time = os.times()[4]
  
  #p = subprocess.Popen([ PROG_BIN , PROG_ARGS ])#stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  p = subprocess.Popen([ PROG_BIN , PROG_ARGS ])#stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  p.wait()
  time = os.times()[4] - start_time
  
  #copy data to a new file
  new_out_filename = currOutputDirName +"/event_" + str(i) + "_" + DEFAULT_OUTFILE_NAME
  out_file = open(new_out_filename, 'w')
  in_file = open(DEFAULT_OUTFILE_NAME, 'r')

  out_file.write('INPUT: ' + PROG_ARGS)
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




