import matplotlib.pyplot as plt
import fine_grained_counters as fgc
from sys import argv
import sniper_stats, sniper_lib
import csv
import os.path 

"""
def write_to_file(dataDir):
	stats = sniper_stats.SniperStats(dataDir)
	marker_id =  1 #TODO: fix the hardcoded value
	total_cores, samples_per_core, counter_stat_dict = fgc.get_sampled_counters(dataDir, stats, marker_id)
	print "total_cores : ", total_cores
	print "samples_per_core: ", samples_per_core
	list_of_perf_tuples = fgc.get_perf_vectors(total_cores, samples_per_core, counter_stat_dict)
	print "dataset size : ", len(list_of_perf_tuples)
	filename = dataDir + "/perf_vector.csv"
	with open(filename, 'wb') as out:
		csv_out=csv.writer(out)
		for row in list_of_perf_tuples:
			csv_out.writerow(row)
	return list_of_perf_tuples
	
def read_from_file(dataDir):
	filename = dataDir + "/perf_vector.csv"
	list_of_perf_tuples = []
	with open(filename, 'r') as infile:
		for line in infile:
			#list_of_perf_tuples.append(fgc.SampleFalseSharing(line.strip().split(',')))
			sampleTuple = tuple([float(x) for x in line.strip().split(',')])
			list_of_perf_tuples.append(fgc.SampleFalseSharing._make(sampleTuple))
			#list_of_perf_tuples.append(fgc.SampleFalseSharing(*sampleTuple))
	return list_of_perf_tuples
"""

"""
command : python plot.py datadir1 datadir2 ...

If perf vector is already in there (in file "perf_vector.csv") 
	read this file and do processing
Otherwise write the csv file and then do processing

"""

def plot_main(argv):
	runs = len(argv) 
	marker_id = 2 #TODO: fix this to set as param
	for run in range(1, runs):
		dataDir = argv[run]
		list_of_perf_tuples = None
		filename = fgc.make_perf_vector_csv_filename(dataDir, marker_id)
		if os.path.isfile(filename):
			list_of_perf_tuples = fgc.read_from_file(dataDir, marker_id)
			print "data read ", len(list_of_perf_tuples)
		else:
			list_of_perf_tuples = fgc.write_to_file(dataDir, marker_id)
		plt.hist([x.l1_invalids for x in list_of_perf_tuples])
		plt.title(dataDir.split('/')[-1])
		plt.ylabel("frequency")
		plt.xlabel("l1_invalids")
		filename = dataDir + "/l1_invalids"
		plt.savefig(filename, format='pdf')
		plt.clf()

		plt.hist([x.ipc for x in list_of_perf_tuples])
		plt.title(dataDir.split('/')[-1])
		plt.ylabel("frequency")
		plt.xlabel("ipc")
		filename = dataDir + "/ipc"
		plt.savefig(filename, format='pdf')
		plt.clf()
		#fig = plt.gcf()
	

if __name__ == '__main__':
	plot_main(argv)
	

