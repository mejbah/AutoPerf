import matplotlib as mpl
mpl.use('Agg') #instead of Xserver for png
import matplotlib.pyplot as plt



def plotDataArray( datasetArray, filename ):
  
  fig = plt.figure()
  ax = fig.add_subplot(111)

  x_values = np.arange(datasetArray.shape[0])
  y_values = datasetArray[:,1] #first column

  
  ax.plot(x_values,y_values)
  fig.savefig(filename)

"""
###########main#################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(10))
fig.savefig('temp.png')
"""
