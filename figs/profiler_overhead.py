"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import matplotlib as mpl
mpl.use('Agg') #instead of Xserver for png
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({'font.size': 18})


in_file_name = "profiler_overhead.txt"
names = []
original = []
with_profiler = []
DELIM = ','

with open(in_file_name,'r') as f:
  for line in f.readlines():
    col = line.strip().split(DELIM)
    print col
    names.append(col[0])
    original.append(col[1])      
    with_profiler.append(col[2])



n_groups = len(names)


fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.25

opacity = 0.3

rects1 = plt.bar(index, original, bar_width,
            #     alpha=opacity,
                 color='b',
                 label='original')

rects2 = plt.bar(index + bar_width, with_profiler, bar_width,
             #    alpha=opacity,
                 color='k',
                 label='run with profiler')


#plt.xlabel('Group')
plt.ylabel('Normalized runtime')
#plt.title('Scores by group and gender')
plt.xticks(index + bar_width, names, rotation='vertical')
#plt.legend()
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend(bbox_to_anchor=(1, 1.58), loc=1, borderaxespad=0.)
plt.tight_layout()
fig.savefig("profiler_overhead.pdf", bbox_inches='tight')
#plt.show()
