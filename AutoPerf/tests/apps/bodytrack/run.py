#!/usr/bin/python

import os
import sys
import subprocess
import re

#all_benchmarks = os.listdir('tests')
#all_benchmarks.remove('Makefile')
#all_benchmarks.remove('defines.mk')
#all_benchmarks.sort()

#all_benchmarks = os.listdir('tests')
#all_benchmarks.remove('Makefile')
#all_benchmarks.remove('defines.mk')
#all_configs = ['defaults']
runs = 10
benchmarks=['./bodytrack-nofs', './bodytrack-pthread']

data = {}
try:
	for benchmark in benchmarks:
		data[benchmark] = []
		for n in range(0, runs):
			start_time = os.times()[4]
			print n 
			p=subprocess.Popen([benchmark, '../../datasets/bodytrack/sequenceB_261', '4', '261', '4000', '5', '0', '8'])
			p.wait()
			time = os.times()[4] - start_time
			print time
			data[benchmark].append(time)
except:
	print 'Aborted!'
	
for benchmark in benchmarks:
	print benchmark,
	if benchmark in data and len(data[benchmark]) == runs:
		if len(data[benchmark]) >= 4:
			mean = (sum(data[benchmark])-max(data[benchmark])-min(data[benchmark]))/(runs-2)
		else:
			mean = sum(data[benchmark][config])/runs
		print '\t'+str(mean),
	else:
		print '\tNOT RUN',
	print
