
#number of perf counter/columns
NUMBER_OF_COUNTERS=19

# error threshold for anomaly
THRESHOLD_ERROR=0.5
#THRESHOLD_ERROR=0.1

#List of networks for autoencoder to choose the best one

#EXPERIMENT_EPOCHS = [ (0,5), (5,10), (10,15), (15,20), (20,25), (25,30), (30,40), (40, 50) ] # for boost
#EXPERIMENT_EPOCHS = [ (0,5),(30,40), (40, 50),(5,10), (10,15), (15,20), (20,25), (25,30)  ] # for boost
#EXPERIMENT_EPOCHS = [ 1,2,3,4,5,6,7,8,9,10 ] # for mysql
EXPERIMENT_EPOCHS = [ (8,9), (9,10),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8) ] # for mysql

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
  "exp11" : [ 16, (4,'relu'),('tied',16)],
  "exp12" : [ 16, (6,'relu'), ('tied',16)],
  "exp13" : [ 16, (8,'relu'), ('tied',16)],
  "exp14" : [ 16, (10,'relu'), ('tied',16)],
  "exp15" : [ 16, (12,'relu'), ('tied',16)],
  "exp16" : [ 16, (14,'relu'), ('tied',16)]
}
"""
NETWORK_CONFIGS = {
  
  "exp9" : [ 16, (12,'relu'), (8,'relu'), ( 4,'relu'), ('tied', 8,'relu'), ('tied', 12,'relu'),('tied',16)]
}
