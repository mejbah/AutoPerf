"""
Author: Mejbah

Collecting performance counters for sim.stats
"""
import sniper_stats, sniper_lib
from sys import argv
import pprint
import os
import sys
import numpy as np
from sklearn import model_selection
import math
#from sklearn import cross_validation

##import machine learning models
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

"""
dict of counter names that we can get from results['results']
every element is a list of len(no_of_cores)
"""
counters = {
	'performance_model.instruction_count': 0,
	'L1-D.loads' : 1,
	'L1-D.stores' : 2,
	'L1-D.coherency-invalidates' : 3,
	'L2.loads' : 4,
	'L2.stores' : 5,
	'L2.coherency-invalidates' : 6,
	'L3.loads' : 7,
	'L3.stores' : 8,
	'L3.coherency-invalidates': 9,
	'branch_predictor.num-incorrect' : 10,
	'branch_predictor.num-correct' : 11,
	'futex.futex_wait_count' : 12
}



"""
use unique marker id for each unique static code block
make unique id using core number for each core accessing the block
increment counter per marker per core
"""

def get_number_of_markers(stats):
	markers =  max( [int(name.split('-')[1]) for name in stats.get_snapshots() if name.startswith('marker')] )
	return markers


"""
collect sampled counters
@return: 
	counter_stat_dict : dict per core, list of dict of counters  for each core
	cores : no of cores
	sample_counts_per_core: dict of list len for each cores
"""
def get_sampled_counters(resultsdir, stats, marker_id):
	counter_stat_dict = {}
	for counter in list(counters.keys()):
		counter_stat_dict[counter] = []

	cores =  max( [int(name.split('-')[2]) for name in stats.get_snapshots() if name.startswith('marker-%d' % marker_id)] )
	sample_counts_per_core = {}
	for core in range(1,cores+1): #core id starts from 0
		niters = max( [ int(name.split('-')[-1]) for name in stats.get_snapshots() if name.startswith('marker-%d-%d' % (marker_id, core))] )
		sample_counts_per_core[core] = niters
		counter_stat_dict[core] = []  # al list of samples for each core, each sample is a dict
		for i in range(1, niters+1): # count starts from 1
			snapshot = sniper_lib.get_results(resultsdir=resultsdir, partial = ('marker-%d-%d-begin-%d' % (marker_id, core, i), 'marker-%d-%d-end-%d' % (marker_id, core, i)))['results']
			sample_dict = {}
			for counter in list(counters.keys()):
				sample_dict[counter] = snapshot[counter][core]  # TODO: only current cores data is collected for specific sample
			counter_stat_dict[core].append(sample_dict)

		print "marker %d : core %d" % (marker_id, core)
	# no of cores == cores as it starts from 1, for parsec we skip the main thread 0
	return (cores, sample_counts_per_core, counter_stat_dict)



def collect_counters(resultsdir, stats, marker_id):
	counter_stat_dict = {}
	for counter in list(counters.keys()):
		counter_stat_dict[counter] = []

	cores =  max( [int(name.split('-')[2]) for name in stats.get_snapshots() if name.startswith('marker-%d' % marker_id)] )
	for core in range(1,cores+1): #core id starts from 0
		niters = max( [ int(name.split('-')[-1]) for name in stats.get_snapshots() if name.startswith('marker-%d-%d' % (marker_id, core))] )
		sum_of_counter_value = {}
		for i in range(1, niters+1): # count starts from 1
			snapshot = sniper_lib.get_results(resultsdir=resultsdir, partial = ('marker-%d-%d-begin-%d' % (marker_id, core, i), 'marker-%d-%d-end-%d' % (marker_id, core, i)))['results']
			for counter in list(counters.keys()):
				if counter in sum_of_counter_value:	
					sum_of_counter_value[counter] += snapshot[counter][core] ##TODO:collecting only the one core value for this snapshot
				else:
					sum_of_counter_value[counter] = snapshot[counter][core]
		# append counter value for this core 
		for counter in list(counters.keys()):
			counter_stat_dict[counter].append(sum_of_counter_value[counter])

		print "marker %d : core %d" % (marker_id, core)
	# no of cores == cores as it starts from 1, for parsec we skip the main thread 0
	return (cores, counter_stat_dict)



def get_perf_vectors(total_cores, samples_per_core, cores_counter_stat_dict):
	#perf_vector = [] * total_cores
	perf_vectors = [] 
	for core in cores_counter_stat_dict:
		for counter_stat_dict in cores_counter_stat_dict[core]:
			total_inst = float(counter_stat_dict['performance_model.instruction_count'])
			#total_L1_access = float(counter_stat_dict['L1-D.loads'] + counter_stat_dict['L1-D.stores'])
			#l1_invalids_per_access = counter_stat_dict['L1-D.coherency-invalidates'] / total_L1_access	
			l1_invalids_pki = (counter_stat_dict['L1-D.coherency-invalidates'] * 1000) / total_inst
			l2_invalids_pki = (counter_stat_dict['L2.coherency-invalidates'] * 1000) / total_inst
			l3_invalids_pki = (counter_stat_dict['L3.coherency-invalidates'] * 1000) / total_inst
			branch_incorrect = counter_stat_dict['branch_predictor.num-incorrect']
			branch_correct = counter_stat_dict['branch_predictor.num-correct']
			branch_factor = float(branch_incorrect) / (branch_correct + branch_incorrect)
			futex_wait_count = counter_stat_dict['futex.futex_wait_count']
			futex_wait_count_pki = (futex_wait_count * 1000) / total_inst
		
			#perf_vector.append((l1_invalids_per_access, branch_factor, futex_wait_count))
			perf_vectors.append((total_cores, l1_invalids_pki, l2_invalids_pki, l3_invalids_pki, branch_factor, futex_wait_count_pki))
		
	return perf_vectors	


def create_perf_vector(total_cores, counter_stat_dict):
	#perf_vector = [] * total_cores
	perf_vector = [] 
	for count in range(total_cores):
		total_inst = float(counter_stat_dict['performance_model.instruction_count'][count])
		#total_L1_access = float(counter_stat_dict['L1-D.loads'][count] + counter_stat_dict['L1-D.stores'][count])
		#l1_invalids_per_access = counter_stat_dict['L1-D.coherency-invalidates'][count] / total_L1_access	
		l1_invalids_pki = (counter_stat_dict['L1-D.coherency-invalidates'][count] * 1000) / total_inst
		l2_invalids_pki = (counter_stat_dict['L2.coherency-invalidates'][count] * 1000) / total_inst
		l3_invalids_pki = (counter_stat_dict['L3.coherency-invalidates'][count] * 1000) / total_inst
		branch_incorrect = counter_stat_dict['branch_predictor.num-incorrect'][count]
		branch_correct = counter_stat_dict['branch_predictor.num-correct'][count]
		branch_factor = float(branch_incorrect) / (branch_correct + branch_incorrect)
		futex_wait_count = counter_stat_dict['futex.futex_wait_count'][count]
		futex_wait_count_pki = (futex_wait_count * 1000) / total_inst
		
		#perf_vector.append((l1_invalids_per_access, branch_factor, futex_wait_count))
		perf_vector.append((total_cores, l1_invalids_pki, l2_invalids_pki, l3_invalids_pki, branch_factor, futex_wait_count_pki))
		
	return perf_vector	
	

"""
cross validate the classifier
return array of scores
"""
def cross_validate(classifier, trainingData, trainingLabels):
	n_splits = 3
	shuffle = True # shuffle the data
	random_state = 0 #TODO: how to select this random_state number
	cv = model_selection.KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
	scores = model_selection.cross_val_score(classifier, trainingData, trainingLabels, cv=cv)
	return scores

if __name__ == '__main__':	

	if len(argv) != 5 :
		print "Usage: counter.py [path_of_train_data_pos] [path_of_train_data_neg] [path_of_test_data] [marker_id]"
		sys.exit(1)
	trainDirPosData = argv[1] ## with bug
	trainDirNegData = argv[2] ## without bug
	testdataDir = argv[3]
	marker_id = int(argv[4])

	
	pp = pprint.PrettyPrinter(indent=4)

	trainingData = [] ##tuples of train data
	trainingLabels = [] ## lables for each tuple in trainingData

	#prepare positive labeled data

	results_for_training = os.listdir(trainDirPosData)
	
	for resultsdir in results_for_training:
			
		stats = sniper_stats.SniperStats(trainDirPosData+ "/" +resultsdir)
		no_of_markers = get_number_of_markers(stats)
		assert marker_id > 0 and marker_id < no_of_markers
		
		#total_cores, counter_stat_dict = collect_counters(trainDirPosData+ "/" +resultsdir, stats, marker_id)
		total_cores, samples_per_core, counter_stat_dict = get_sampled_counters(trainDirPosData+ "/" +resultsdir, stats, marker_id)
	
		# get input vector created using normalized results 
		#list_of_perf_tuples = create_perf_vector(total_cores, counter_stat_dict)
		list_of_perf_tuples = get_perf_vectors(total_cores, samples_per_core, counter_stat_dict)

		# label the pos data with value '1'
		list_of_labels = [1] * len(list_of_perf_tuples)

		trainingData.extend(list_of_perf_tuples)
		trainingLabels.extend(list_of_labels)

	
	#prepare negative labeled data
	
	results_for_training = os.listdir(trainDirNegData)

	for resultsdir in results_for_training:

		stats = sniper_stats.SniperStats(trainDirNegData+ "/" +resultsdir)
		no_of_markers = get_number_of_markers(stats)
		assert marker_id > 0 and marker_id < no_of_markers
		
		#total_cores, counter_stat_dict = collect_counters(trainDirNegData+ "/" +resultsdir, stats, marker_id)
		total_cores, samples_per_core, counter_stat_dict = get_sampled_counters(trainDirNegData+ "/" +resultsdir, stats, marker_id)

		#list_of_perf_tuples = create_perf_vector(total_cores, counter_stat_dict)
		list_of_perf_tuples = get_perf_vectors(total_cores, samples_per_core, counter_stat_dict)

		#lable the neg data with value '-1'
		list_of_labels = [-1] * len(list_of_perf_tuples)

		trainingData.extend(list_of_perf_tuples)
		trainingLabels.extend(list_of_labels)

	#print len(trainingData)
	#pp.pprint(trainingData)
	
	## get a classifier		
	##########################	
	## Support vector machine
	##########################
	kernels = ['linear','rbf', 'sigmoid']
	C = 1.0  # SVM regularization parameter
	max_avg_score = 0.0
	selected_kernel = ""

	for kernel in kernels:	
		
		classifier = SVC(kernel=kernel, C=C)

		## cross validation
		scores = cross_validate(classifier, trainingData, trainingLabels)
		#pp.pprint(scores)

		mean = np.mean(scores)
		if mean > max_avg_score:
			max_avg_score = mean
			selected_kernel = kernel
		
	print "SVC with kernel -" , selected_kernel , "- score : ", max_avg_score 	
	

	#########################
	## Perceptron
	#########################
	classifier = Perceptron()			
	
	scores = cross_validate(classifier, trainingData, trainingLabels)	
	
	max_avg_score = np.mean(scores)

	print "Perceptron score : " , max_avg_score

	##########################
	## Decisiontree classifier
	##########################
	classifier = DecisionTreeClassifier(random_state=0)
	
	scores = cross_validate(classifier, trainingData, trainingLabels)
		
	max_avg_score = np.mean(scores)
	
	print "DecisionTreeClassifier scores : " , max_avg_score


	###########################
	## MLP
	###########################
	
	# solver 'adam' works well for large datasets
	# for small datesets 'lbfgs' is fast converging

	##TODO: search for diffetent solver and different networks
	activation = 'relu'
	solver = 'lbfgs'
	hidden_layer_sizes = ( 20, 20 ) # tuple of hidden layer sizes, each entry in # of neuron in each hidden layer
	alpha= 1 * math.exp(-5) # regularization paramater
	classifier = MLPClassifier(activation=activation, solver= solver, hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, random_state=1)

	scores = cross_validate(classifier, trainingData, trainingLabels)
	
	max_avg_score = np.mean(scores)

	print "MLP scores : ", max_avg_score
	
	## train
	#classifier.fit(np.array(trainingData), np.array(trainingLabels))

	## test
	#results_for_testing = os.listdir(testdataDir)

	#for resultsdir in results_for_testing:
	#	total_cores, counter_stat_dict = collect_counters(testdataDir + "/" + resultsdir)
	#	list_of_perf_tuples = create_perf_vector(total_cores, counter_stat_dict)

		
