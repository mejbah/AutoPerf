"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import matplotlib as mpl
mpl.use('Agg') #instead of Xserver for png
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({'font.size': 20})
#Number of clusters,False positive rate,F1 score
in_file_name = "sqlite_cluster.txt"
names = []
fprate = []
f1 = []
DELIM = ','

with open(in_file_name,'r') as f:
  for line in f.readlines():
    col = line.strip().split(DELIM)
    print col
    names.append(col[0])
    fprate.append(col[1])      
    f1.append(col[2])


n_groups = len(names)


fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.15

opacity = 0.3

rects1 = plt.bar(index, fprate, bar_width,
            #     alpha=opacity,
                 color='b',
                 label='False positive rate')

rects2 = plt.bar(index + bar_width, f1, bar_width,
             #    alpha=opacity,
                 color='r',
                 label='F1 score')


plt.xlabel('Number of cluster')
#plt.ylabel('F1 score')
#plt.title('Scores by group and gender')
#plt.xticks(index + bar_width, names, rotation='vertical')
plt.xticks(index + bar_width/2, names)
#plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
fig.savefig("sqlite_cluster.pdf", bbox_inches='tight')
#fig.savefig("sqlite_cluster.pdf")
#plt.show()
