"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import matplotlib as mpl
mpl.use('Agg') #instead of Xserver for png
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({'font.size': 14})

in_file_name = "kmeans_kernels_k2.txt"
names = []
precision = []
recall = []
f1score = []
DELIM = ','

with open(in_file_name,'r') as f:
  for line in f.readlines():
    col = line.strip().split(DELIM)
    print col
    names.append(col[0])
    precision.append(col[1])      
    recall.append(col[2])
    f1score.append(col[3])

n_groups = len(names)


fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.15

opacity = 0.3

t = np.arange(0.0, 2.0, 0.01)
s1 = np.sin(2*np.pi*t)
s2 = np.sin(4*np.pi*t)

plt.figure(1)
#plt.subplot(211) #two row, one col, first one
##plot first ###
plt.subplot(131)  #one row, two col, first one

rects1 = plt.bar(index, precision, bar_width,
            #     alpha=opacity,
                 color='b',
                 label='precision')

rects2 = plt.bar(index + bar_width, recall, bar_width,
             #    alpha=opacity,
                 color='r',
                 label='recall')

rects3 = plt.bar(index + bar_width + bar_width, f1score, bar_width,
#                 alpha=opacity,
                 color='c',
                 label='F1 score')



plt.xticks(index + bar_width + (bar_width/2.0), names, rotation='vertical')


###plot second one###
plt.subplot(132)

in_file_name = "kmeans_kernels_k3.txt"
names = []
precision = []
recall = []
f1score = []
DELIM = ','

with open(in_file_name,'r') as f:
  for line in f.readlines():
    col = line.strip().split(DELIM)
    print col
    names.append(col[0])
    precision.append(col[1])      
    recall.append(col[2])
    f1score.append(col[3])

n_groups = len(names)
index = np.arange(n_groups)
rects1 = plt.bar(index, precision, bar_width,
            #     alpha=opacity,
                 color='b',
                 label='precision')

rects2 = plt.bar(index + bar_width, recall, bar_width,
             #    alpha=opacity,
                 color='r',
                 label='recall')

rects3 = plt.bar(index + bar_width + bar_width, f1score, bar_width,
#                 alpha=opacity,
                 color='c',
                 label='F1 score')


plt.xticks(index + bar_width + (bar_width/2.0), names, rotation='vertical')


##### plot 3rd in the fist row###

plt.subplot(133)

in_file_name = "kmeans_kernels_k4.txt"
names = []
precision = []
recall = []
f1score = []
DELIM = ','

with open(in_file_name,'r') as f:
  for line in f.readlines():
    col = line.strip().split(DELIM)
    print col
    names.append(col[0])
    precision.append(col[1])      
    recall.append(col[2])
    f1score.append(col[3])

n_groups = len(names)
index = np.arange(n_groups)

rects1 = plt.bar(index, precision, bar_width,
            #     alpha=opacity,
                 color='b',
                 label='precision')

rects2 = plt.bar(index + bar_width, recall, bar_width,
             #    alpha=opacity,
                 color='r',
                 label='recall')

rects3 = plt.bar(index + bar_width + bar_width, f1score, bar_width,
#                 alpha=opacity,
                 color='c',
                 label='F1 score')





#plt.xlabel('Cluster number')
#plt.ylabel('F1 score')
#plt.title('Scores by group and gender')
#plt.xticks(index + bar_width + (bar_width/2.0), names, rotation='vertical')
plt.xticks(index + bar_width + (bar_width/2.0), names, rotation='vertical')





#plt.legend()
plt.legend(bbox_to_anchor=(1,1.30), loc=1, borderaxespad=0.)
plt.tight_layout()
fig.savefig("kmeans_kernels.pdf", bbox_inches='tight')
#plt.show()
