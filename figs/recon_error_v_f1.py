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

  
  ax.plot(x_values,y_values, 'co',markersize=1)
  
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

 
  ax.legend(loc='upper left') 
  #ax.set_ylabel(label)
  #ax.set_xlabel("Performance Counter Samples for Training(not anomalous)")

  fig.savefig(filename, format='pdf')



def doMain(in_file_name, out_file_name):
  x_vals = []
  y_vals = []
  x_col_index = 0
  y_col_index = 1
  x_label = "Reconstruction Error (RE)"
  y_label = "F1 score"
  DELIM = '\t'
  
  with open(in_file_name,'r') as f:
    for line in f.readlines():
      col = line.strip().split(DELIM)
      print col
      x_vals.append(col[x_col_index])
      y_vals.append(col[y_col_index])      

  dataset = zip(x_vals, y_vals)
  dataset.sort(key=lambda tup: tup[0])  # sorts in place
  x_vals = []
  y_vals = []
  for tup in dataset:
    x_vals.append(tup[0]) 
    y_vals.append(tup[1]) 
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.plot(x_vals, y_vals, '-ko')
  fig.savefig(out_file_name, format='pdf') 
  
if __name__ == "__main__":
  
  mpl.rcParams.update({'font.size': 14})
  infile = sys.argv[1]
  outfile = sys.argv[2]
  doMain(infile, outfile)

