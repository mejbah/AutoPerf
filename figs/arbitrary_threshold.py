"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import matplotlib as mpl
mpl.use('Agg') #instead of Xserver for png
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({'font.size': 14})

in_file_name = "arbitrary_threshold.txt"
names = []
ten = []
twenty = []
forty = []
re = []
DELIM = ','

with open(in_file_name,'r') as f:
  for line in f.readlines():
    col = line.strip().split(DELIM)
    print col
    names.append(col[0])
    ten.append(col[1])      
    twenty.append(col[2])
    forty.append(col[3])
    re.append(col[4])



n_groups = len(names)


fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.15

opacity = 0.3

rects1 = plt.bar(index, ten, bar_width,
            #     alpha=opacity,
                 color='b',
                 label='10%')

rects2 = plt.bar(index + bar_width, twenty, bar_width,
             #    alpha=opacity,
                 color='r',
                 label='20%')

rects3 = plt.bar(index + bar_width + bar_width, forty, bar_width,
#                 alpha=opacity,
                 color='c',
                 label='40%')

rects3 = plt.bar(index + 3*bar_width, re, bar_width,
#                 alpha=opacity,
                 color='k',
                 label='RE')

#plt.xlabel('Group')
plt.ylabel('F1 score')
#plt.title('Scores by group and gender')
plt.xticks(index + bar_width, names, rotation='vertical')
#plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
fig.savefig("arbitrary_threshold.pdf", bbox_inches='tight')
#plt.show()
