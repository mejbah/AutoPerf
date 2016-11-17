"""
Author: Mejbah

Collecting performance counters for sim.stats
"""
import sniper_lib
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
collect counters from stat and return a dict of list for each counter
"""
def collect_counters(resultsdir):

	print "Reading data from: " + resultsdir + "\n"

	results = sniper_lib.get_results(resultsdir=resultsdir)

	total_cores = int(results['config']['general/total_cores'])


	counter_stat_dict = {}


	for counter in list(counters.keys()):
		counter_stat_dict[counter] = []
		for x in range(total_cores):
			counter_stat_dict[counter].append(results['results'][counter][x])
			#print results['results']['L1-D.loads'][x]

	return (total_cores, counter_stat_dict)



def create_perf_vector(total_cores, counter_stat_dict):
	#perf_vector = [] * total_cores
	perf_vector = [] 
	for core in range(total_cores):
		total_inst = float(counter_stat_dict['performance_model.instruction_count'][core])
		#total_L1_access = float(counter_stat_dict['L1-D.loads'][core] + counter_stat_dict['L1-D.stores'][core])
		#l1_invalids_per_access = counter_stat_dict['L1-D.coherency-invalidates'][core] / total_L1_access	
		l1_invalids_pki = (counter_stat_dict['L1-D.coherency-invalidates'][core] * 1000) / total_inst
		branch_incorrect = counter_stat_dict['branch_predictor.num-incorrect'][core]
		branch_correct = counter_stat_dict['branch_predictor.num-correct'][core]
		branch_factor = float(branch_incorrect) / (branch_correct + branch_incorrect)
		futex_wait_count = counter_stat_dict['futex.futex_wait_count'][core]
		futex_wait_count_pki = (futex_wait_count * 1000) / total_inst
		
		#perf_vector.append((l1_invalids_per_access, branch_factor, futex_wait_count))
		perf_vector.append((total_cores, l1_invalids_pki, branch_factor, futex_wait_count_pki))
		
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

	if len(argv) != 4 :
		print "Usage: counter.py [path_of_train_data_pos] [path_of_train_data_neg] [path_of_test_data]"
		sys.exit(1)
	trainDirPosData = argv[1] ## with bug
	trainDirNegData = argv[2] ## without bug
	testdataDir = argv[3]

	
	pp = pprint.PrettyPrinter(indent=4)

	trainingData = [] ##tuples of train data
	trainingLabels = [] ## lables for each tuple in trainingData

	#prepare positive labeled data

	results_for_training = os.listdir(trainDirPosData)
	
	for resultsdir in results_for_training:
		
		total_cores, counter_stat_dict = collect_counters(trainDirPosData+ "/" +resultsdir)
		
		#pp.pprint(counter_stat_dict)
	
		# get input vector created using normalized results 
		list_of_perf_tuples = create_perf_vector(total_cores, counter_stat_dict)
		# label the pos data with value '1'
		list_of_labels = [1] * len(list_of_perf_tuples)
		trainingData.extend(list_of_perf_tuples)
		trainingLabels.extend(list_of_labels)

	
	#prepare negative labeled data
	
	results_for_training = os.listdir(trainDirNegData)

	for resultsdir in results_for_training:
		total_cores, counter_stat_dict = collect_counters(trainDirNegData + "/" + resultsdir)	
		list_of_perf_tuples = create_perf_vector(total_cores, counter_stat_dict)
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

		
