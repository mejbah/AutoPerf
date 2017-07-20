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
from collections import namedtuple
#from sklearn import cross_validation

##import machine learning models
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import csv
import os.path 




##import plot lib
import matplotlib.pyplot as plt


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
	'futex.futex_wait_count' : 12,
	'performance_model.cycle_count': 13,
	'L1-D.mshr-latency' : 14,
	'L2.mshr-latency' : 15,
	'L3.mshr-latency' : 16
}

SampleFalseSharing = namedtuple('SampleFalseSharing', ['total_cores', 'l1_invalids', 'l2_invalids', 'l3_invalids', 'branch_factor', 'ipc'])

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
	for core in range(1,cores+1): #core id starts from 0 TODO: fix for core 0 only data
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
			ipc = total_inst / counter_stat_dict['performance_model.cycle_count'] #TODO: when and why to use ipc? for checking/deciding if the perf vector should be labeled positive(perf bug) or negtive(no perf bug)
		
			#perf_vectors.append((total_cores, l1_invalids_pki, l2_invalids_pki, l3_invalids_pki, branch_factor, futex_wait_count_pki))
			perf_vectors.append(SampleFalseSharing(total_cores, l1_invalids_pki, l2_invalids_pki, l3_invalids_pki, branch_factor, ipc))
		
	return perf_vectors	



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


"""
predict label for test data using the classifier model
@return: array of labels
"""
def predict_test_data(classifier, testData):
	return classifier.predict(testData)


"""
decide predicted label based on the labels predicting on test data
TODO; for now using sum of all +1 and -1 labels
"""
def get_predicted_class(predictedLables):
	predicted_label = np.sum(predictedLabels)
	return predicted_label


def write_to_file(dataDir, marker_id):
	stats = sniper_stats.SniperStats(dataDir)
	
	no_of_markers = get_number_of_markers(stats)
	print "markers in application" ,  no_of_markers
	assert marker_id > 0 and marker_id <= no_of_markers
	#marker_id =  1 #TODO: fix the hardcoded value
	total_cores, samples_per_core, counter_stat_dict = get_sampled_counters(dataDir, stats, marker_id)
	print "total_cores : ", total_cores
	print "samples_per_core: ", samples_per_core
	list_of_perf_tuples = get_perf_vectors(total_cores, samples_per_core, counter_stat_dict)
	print "dataset size : ", len(list_of_perf_tuples)
	filename = dataDir + "/perf_vector_marker_" + str(marker_id) + ".csv"
	with open(filename, 'wb') as out:
		csv_out=csv.writer(out)
		for row in list_of_perf_tuples:
			csv_out.writerow(row)
	return list_of_perf_tuples
	

def read_from_file(dataDir, marker_id):
	filename = make_perf_vector_csv_filename(dataDir, marker_id)
	list_of_perf_tuples = []
	with open(filename, 'r') as infile:
		for line in infile:
			sampleTuple = tuple([float(x) for x in line.strip().split(',')])
			list_of_perf_tuples.append(SampleFalseSharing._make(sampleTuple))
			#list_of_perf_tuples.append(SampleFalseSharing(*sampleTuple))
	return list_of_perf_tuples


def get_list_of_perf_tuples(dataDir, marker_id):
	filename = make_perf_vector_csv_filename(dataDir, marker_id)
	if os.path.isfile(filename):
		return read_from_file(dataDir, marker_id)
	else:
		return write_to_file(dataDir, marker_id)

def make_perf_vector_csv_filename(dataDir, marker_id):
	filename = dataDir + "/perf_vector_marker_" + str(marker_id) +".csv"
	return filename


if __name__ == '__main__':	

	if len(argv) < 5 :
		print "Usage: counter.py [path_of_train_data_pos] [path_of_train_data_neg] [path_of_test_data] [marker_id] [out_file]"
		sys.exit(1)
	trainDirPosData = argv[1] ## with bug
	trainDirNegData = argv[2] ## without bug
	testdataDir = argv[3]
	marker_id = int(argv[4])
	out_file = "summary.out"
	if len(argv) > 5:
		out_file = argv[5]

	f_out = open(out_file, 'w')
	
	print >> f_out, "Training :", trainDirPosData, " + ", trainDirNegData
	print >> f_out, "Test: " , testdataDir
	
	pp = pprint.PrettyPrinter(indent=4)

	trainingData = [] ##tuples of train data
	trainingLabels = [] ## lables for each tuple in trainingData

	#prepare positive labeled data

	results_for_training = os.listdir(trainDirPosData)
	
	for resultsdir in results_for_training:
		"""		
		stats = sniper_stats.SniperStats(trainDirPosData+ "/" +resultsdir)
		no_of_markers = get_number_of_markers(stats)
		assert marker_id > 0 and marker_id < no_of_markers
		
		total_cores, samples_per_core, counter_stat_dict = get_sampled_counters(trainDirPosData+ "/" +resultsdir, stats, marker_id)
	
		# get input vector created using normalized results 
		list_of_perf_tuples = get_perf_vectors(total_cores, samples_per_core, counter_stat_dict)
		"""
		list_of_perf_tuples = get_list_of_perf_tuples(trainDirPosData+ "/" +resultsdir, marker_id)

		# label the pos data with value '1'
		list_of_labels = [1] * len(list_of_perf_tuples)

		trainingData.extend(list_of_perf_tuples)
		trainingLabels.extend(list_of_labels)

	
	#prepare negative labeled data
	
	results_for_training = os.listdir(trainDirNegData)

	for resultsdir in results_for_training:
		"""
		stats = sniper_stats.SniperStats(trainDirNegData+ "/" +resultsdir)
		no_of_markers = get_number_of_markers(stats)
		assert marker_id > 0 and marker_id < no_of_markers
		
		total_cores, samples_per_core, counter_stat_dict = get_sampled_counters(trainDirNegData+ "/" +resultsdir, stats, marker_id)

		list_of_perf_tuples = get_perf_vectors(total_cores, samples_per_core, counter_stat_dict)
		"""
		list_of_perf_tuples = get_list_of_perf_tuples(trainDirNegData+ "/" +resultsdir, marker_id)

		#lable the neg data with value '-1'
		list_of_labels = [-1] * len(list_of_perf_tuples)

		trainingData.extend(list_of_perf_tuples)
		trainingLabels.extend(list_of_labels)


	## prepare testing labeled data
	testData = []

	results_for_testing = os.listdir(testdataDir)

	for resultsdir in results_for_testing:
		"""	
		stats = sniper_stats.SniperStats(testdataDir+ "/" +resultsdir)
		no_of_markers = get_number_of_markers(stats)
		assert marker_id > 0 and marker_id < no_of_markers

		total_cores, samples_per_core, counter_stat_dict = get_sampled_counters(testdataDir + "/" +resultsdir, stats, marker_id)
		
		list_of_perf_tuples = get_perf_vectors(total_cores, samples_per_core, counter_stat_dict)
		"""
		list_of_perf_tuples = get_list_of_perf_tuples(testdataDir + "/" +resultsdir, marker_id)
		
		testData.extend(list_of_perf_tuples)
		
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
	print >> f_out, "SVC with kernel -" , selected_kernel , "- score : ", max_avg_score 	
	
	## train/fit model
	clf = SVC(kernel=selected_kernel, C=C)
	clf.fit(trainingData, trainingLabels)

	## test with fitted model
	predictedLabels = predict_test_data(clf, testData)	
	
	predicted_label	= get_predicted_class(predictedLabels)
	
	print "Predicted label ", predicted_label
	print >> f_out, "Predicted label ", predicted_label
		
	#########################
	## Perceptron
	#########################
	classifier = Perceptron()			
	
	scores = cross_validate(classifier, trainingData, trainingLabels)	
	
	max_avg_score = np.mean(scores)

	print "Perceptron score : " , max_avg_score
	print >> f_out, "Perceptron score : " , max_avg_score

	## train/fit model
	classifier.fit(trainingData, trainingLabels)

	## test with fitted model
	predictedLabels = predict_test_data(clf, testData)	
	
	predicted_label	= get_predicted_class(predictedLabels)

	print "Predicted label ", predicted_label
	print >> f_out, "Predicted label ", predicted_label

	##########################
	## Decisiontree classifier
	##########################
	classifier = DecisionTreeClassifier(random_state=0)
	
	scores = cross_validate(classifier, trainingData, trainingLabels)
		
	max_avg_score = np.mean(scores)
	
	print "DecisionTreeClassifier scores : " , max_avg_score
	print >> f_out,  "DecisionTreeClassifier scores : " , max_avg_score

	## train/fit model
	classifier.fit(trainingData, trainingLabels)

	## test with fitted model
	predictedLabels = predict_test_data(clf, testData)	
	
	predicted_label	= get_predicted_class(predictedLabels)
	
	print "Predicted label ", predicted_label
	print >> f_out, "Predicted label ", predicted_label

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
	print >> f_out, "MLP scores : ", max_avg_score

	## train/fit model
	classifier.fit(trainingData, trainingLabels)

	## test with fitted model
	predictedLabels = predict_test_data(clf, testData)	
	
	predicted_label	= get_predicted_class(predictedLabels)
	
	print "Predicted label ", predicted_label
	print >> f_out, "Predicted label ", predicted_label
		
