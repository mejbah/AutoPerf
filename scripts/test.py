from utils import *
from sys import argv

if __name__ == "__main__" :
  dataDirGood = argv[1]
  dataDirBad = argv[2]
  counterId = argv[3] 
  print compareCounters(dataDirGood, dataDirBad, counterId)
