import math
from concurrent.futures import process
from hashlib import new
from multiprocessing.resource_sharer import stop
import docx2txt #lib for reading docx files
import re
import pandas as pd
import numpy as np
import csv as csv
import random
#for data set
trainingSet = []
testSet = []    
newDataset = []


#splits the data set to training set and testset
def dataSet(filename, split,trainingSet = [], testSet=[]): 
    with open(filename,'r') as csvFile:
        lines = csv.reader(csvFile)
        dataset = list(lines)
        print(dataset)
        print("\n")
        for x in range(len(dataset)-1):
            for y in range(4):
                if random.random() < split:
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])
        
#second step in KNN algorithm
#hamming,Minkowski,Manhattan distance are options but euclidean is the most common 
#euclidean formula = d = sqrt(pow(x-y)^2)
def getEuclideanDistance (pointOne, pointTwo, length):
    distance = 0
    for x in range(length):
        distance += pow((pointOne[x] - pointTwo[x]),2)
    return math.sqrt(distance)
def getNeighbors():
    print("data set")
def getResponse():
    print("getting the votes made by the neighbors")
def getAccuracy():
    print("accuracy of the model")

data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = getEuclideanDistance(data1, data2, 3)
dataSet(r'finalCSV.csv',0.66,trainingSet,testSet)
print('Training Set: ' + repr(len(trainingSet)))
print('Test Set: ' + repr(len(testSet)))

