from __future__ import division
from sys import argv
import numpy
from scipy import stats

"""
1. read file:each line(except first 3) has the following format:
	markid, threadCount, instruction count, eventcount
2. perform stats
"""

inFile = argv[1]
runInfo = ""
counterName = None
normalizedEventCount = []

with open(inFile,'r') as fp:
  for linenumber,line in enumerate(fp):
	if linenumber < 2:
	  runInfo += line
	  runInfo += '\n' 
  
	elif linenumber == 2:
	  counterName = line.strip().split(",")[-1]

	else:
	  colValues = line.strip().split(",")
	  instructionCounter = int(colValues[2])
	  currCounter = int(colValues[3])
	  normalizedEventCount.append((currCounter/instructionCounter))

	  

#do stats
print runInfo
print counterName
print stats.hmean(normalizedEventCount)
print numpy.var(normalizedEventCount)
print numpy.std(normalizedEventCount)


	  


