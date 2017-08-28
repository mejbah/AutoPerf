from __future__ import division
import matplotlib as mpl
mpl.use('Agg') #instead of Xserver for png
import matplotlib.pyplot as plt
import sys
import numpy


def plotDataList( datasetList, filename, label="default", secondDatasetList=None, thirdDatasetList=None  ):
  
  fig = plt.figure()
  ax = fig.add_subplot(111)

  x_values = [x for x in range(len(datasetList))]
  y_values = datasetList 

   
  markers_on = [58]
  #ax.plot(x_values,y_values, '-o',markersize=1)
  ax.plot(x_values,y_values, '-kD', markevery=markers_on)
  
  if secondDatasetList != None:

    x_values = [x for x in range(len(secondDatasetList))]
    y_values = secondDatasetList
    dashes = [10, 5, 10, 5]  # 10 points on, 5 off, 100 on, 5 off 
    #ax.plot(x_values, y_values, 'ro', label = 'Thresold Error')
    ax.plot(x_values, y_values, 'r-',dashes=dashes, label = 'Threshold Error')
    #ax.plot(x_values, y_values, 'r-',dashes=dashes)

  if thirdDatasetList != None:

    x_values = [x for x in range(len(thirdDatasetList))]
    y_values = thirdDatasetList

  
    ax.plot(x_values, y_values, 'g^', label = 'third')

 
  #ax.legend(loc='upper left') 
  ax.set_ylabel("RE")
  ax.set_xlabel("Number of runs for training")
  
  plt.tight_layout()
  fig.savefig(filename)








mpl.rcParams.update({'font.size': 14})

infile = "errors_for_mysql_training_dataset.txt" #sys.argv[1]

numbers = []
with open(infile) as fs:
  for line in fs.readlines():
    line = line.strip()
    numbers.append(float(line))


nsamples = len(numbers)


for i in range(nsamples):
  numbers[i] = (numbers[i] + 2) / 200.0 + 0.005

plotDataList( numbers, "variance_termination_mysql.pdf",  "Reconstruction Error")


