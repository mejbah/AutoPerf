import matplotlib as mpl
mpl.use('Agg') #instead of Xserver for png
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({'font.size': 20})
##Cluster,Apache,Postgres,Mesos,MySQL
in_file_name = "cluster_false_positive_rates.txt"
names = []
apache = []
postgres = []
mesos = []
mysql = []
DELIM = ','

with open(in_file_name,'r') as f:
  for line in f.readlines():
    col = line.strip().split(DELIM)
    print col
    names.append(col[0])
    apache.append(col[1])      
    mesos.append(col[2])
    postgres.append(col[3])
    mysql.append(col[4])



n_groups = len(names)


fig, ax = plt.subplots()

index = np.arange(n_groups)
# red dashes, blue squares and green triangles
plt.plot(names, apache, 'ro-',label='Apache')
plt.plot(names, mesos, 'bs-',label='Mesos')
plt.plot(names, postgres, 'g^-',label='Postgres')
plt.plot(names, mysql, 'k+-',label='MySQL')

plt.xticks(index+1, names)
ax.legend(loc='upper right') 
ax.set_xlabel("Number of cluster")
ax.set_ylabel("False positive rate")

#plt.legend()
#plt.tight_layout()
fig.savefig("cluster_false_positive_rates.pdf", bbox_inches='tight')
