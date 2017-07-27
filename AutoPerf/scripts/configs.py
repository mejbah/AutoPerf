
#number of perf counter/columns
NUMBER_OF_COUNTERS= 12 #16 #19 #33
#SCALE_UP_FACTOR=1000000
SCALE_UP_FACTOR=1000

# error threshold for anomaly
#THRESHOLD_ERROR=0.5
THRESHOLD_ERROR=0.2

#List of networks for autoencoder to choose the best one

#EXPERIMENT_EPOCHS = [ (0,5), (5,10), (10,15), (15,20), (20,25), (25,30), (30,40), (40, 50) ] # for boost
#EXPERIMENT_EPOCHS = [ (0,5),(30,40), (40, 50),(5,10), (10,15), (15,20), (20,25), (25,30)  ] # for boost
#EXPERIMENT_EPOCHS = [ 1,2,3,4,5,6,7,8,9,10 ] # for mysql
#EXPERIMENT_EPOCHS = [ (8,9), (9,10),(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8) ] # for mysql
#EXPERIMENT_EPOCHS = [ (9,10),(5,6),(6,7), (0,1),(1,2),(2,3),(3,4),(4,5),(7,8), (8,9) ] # for mysql

"""
NETWORK_CONFIGS = {
  "exp1" : [ 16, (12,'relu'), ( 10,'relu'), ('tied', 12,'relu'), ('tied',16)],
  "exp2" : [ 16, (12,'relu'), ( 8,'relu'), ('tied', 12,'relu'), ('tied',16)],
  "exp3" : [ 16, (12,'relu'), ( 6,'relu'), ('tied', 12,'relu'), ('tied',16)],
  "exp4" : [ 16, (12,'relu'), ( 4,'relu'), ('tied', 12,'relu'), ('tied',16)],

  "exp5" : [ 16, (8,'relu'), ( 6,'relu'), ('tied', 8,'relu'), ('tied',16)],
  "exp6" : [ 16, (8,'relu'), ( 4,'relu'), ('tied', 8,'relu'), ('tied',16)],
  "exp7" : [ 16, (8,'relu'), ( 2,'relu'), ('tied', 8,'relu'), ('tied',16)],

  "exp8" : [ 16, (12,'relu'), (8,'relu'), ( 6,'relu'), ('tied', 8,'relu'), ('tied', 12,'relu'),('tied',16)],
  "exp9" : [ 16, (12,'relu'), (8,'relu'), ( 4,'relu'), ('tied', 8,'relu'), ('tied', 12,'relu'),('tied',16)],
  "exp10" : [ 16, (14,'relu'), ( 12,'relu'), ('tied', 14,'relu'), ('tied',16)],

  "exp11" : [ 16, (8,'relu'),('tied',16)],
  "exp12" : [ 16, (9,'relu'), ('tied',16)],
  "exp13" : [ 16, (10,'relu'), ('tied',16)],
  "exp14" : [ 16, (11,'relu'), ('tied',16)],
  "exp15" : [ 16, (12,'relu'), ('tied',16)],
  "exp16" : [ 16, (13,'relu'), ('tied',16)],
  "exp17" : [ 16, (14,'relu'), ('tied',16)],
  "exp18" : [ 16, (15,'relu'), ('tied',16)]
}

"""
"""
NETWORK_CONFIGS = {

#  "exp11" : [ 16, (8,'relu'),('tied',16)],
#  "exp12" : [ 16, (9,'relu'), ('tied',16)],
#  "exp13" : [ 16, (10,'relu'), ('tied',16)],
#  "exp14" : [ 16, (11,'relu'), ('tied',16)],
#  "exp15" : [ 16, (12,'relu'), ('tied',16)],
#  "exp16" : [ 16, (13,'relu'), ('tied',16)],
#  "exp17" : [ 16, (14,'relu'), ('tied',16)],
#  "exp18" : [ 16, (15,'relu'), ('tied',16)]

### for layer 1  
#  "exp1" : [ 11, (9,'relu'), ('tied',11)],
#  "exp2" : [ 11, (8,'relu'), ('tied',11)],
#  "exp3" : [ 11, (7,'relu'), ('tied',11)],
#  "exp4" : [ 11, (6,'relu'), ('tied',11)],
  "exp5" : [ 11, (5,'relu'), ('tied',11)],
#  "exp6" : [ 11, (10,'relu'), ('tied',11)]
}
"""

"""
NETWORK_CONFIGS = {

  #for layer 2
  "exp1" : [ 11, (9,'relu'), (8, 'relu'),('tied',9), ('tied',11)],
  "exp2" : [ 11, (9,'relu'), (7, 'relu'),('tied',9), ('tied',11)],
  "exp3" : [ 11, (9,'relu'), (6, 'relu'),('tied',9), ('tied',11)],
  "exp4" : [ 11, (9,'relu'), (5, 'relu'),('tied',9), ('tied',11)]

}
"""
NETWORK_CONFIGS = {

  "exp1" : [ 12, (9,'relu'), (8, 'relu'),('tied',9), ('tied',12)],
}

"""
NETWORK_CONFIGS = {

  #for layer 3
#  "exp1" : [ 11, (9,'relu'), (8, 'relu'),(7,'relu'),('tied',8),('tied',9), ('tied',11)],
  "exp2" : [ 11, (9,'relu'), (8, 'relu'),(6,'relu'),('tied',8),('tied',9), ('tied',11)],  #selected best based on error minimization
#  "exp3" : [ 11, (9,'relu'), (8, 'relu'),(5,'relu'),('tied',8),('tied',9), ('tied',11)],
#  "exp4" : [ 11, (9,'relu'), (8, 'relu'),(4,'relu'),('tied',8),('tied',9), ('tied',11)],


}
"""

MODEL_SAVED_FILE_NAME = "trained_network"

def generateHiddenLayerConfigs( inputLen, activation ):
  max = inputLen - 1
  min = inputLen/2
  listOfHiddenNetworks = []
  for hiddenNode in range(min, max+1):
    nextTuple = (hiddenNode,activation)
    listOfHiddenNetworks.append(nextTuple)

  return listOfHiddenNetworks


def getNextKeyID():
  return len(NETWORK_CONFIGS) + 1

def setNetworkConfigs( networkPrefix, networkSuffix, inputLen, startID=None ): #startID can be set using  getNextKeyID() 
  if startID==None:
    NETWORK_CONFIGS.clear() ##clear the exisitng dict
    startID = 1
  newLayers = generateHiddenLayerConfigs( inputLen, 'relu' )
  currID = startID
  for layer in newLayers:
    key = 'exp' + str(currID)
    currID += 1
    newNetwork = []
    newNetwork.extend(networkPrefix)
    newNetwork.append(layer)
    newNetwork.extend(networkSuffix)
    NETWORK_CONFIGS[key] = newNetwork



def getEncodeLen(prefix):
    return prefix[-1][0] #first element of last tuple of prefix  

def getNetworkPrefix( networkConfig, numberofLayer ):
  return networkConfig[:numberofLayer]

def getNetworkSuffix( networkConfig, numberOfLayer ):
  suffix = networkConfig[(0-numberOfLayer):]
  middleNodeNumber = suffix[0][0]
  modifiedTuple = ('tied', middleNodeNumber )
  suffix[0] = modifiedTuple
  return suffix

if __name__ == "__main__" :
   inputLen = 16
   networkPrefix = [ inputLen ]
   networkSuffix = [ ('tied', inputLen) ]

   setNetworkConfigs( networkPrefix, networkSuffix, inputLen, 1 )
  
   for key in NETWORK_CONFIGS:
    print key
    print NETWORK_CONFIGS[key]
    print getNetworkSuffix(NETWORK_CONFIGS[key],2)



  

