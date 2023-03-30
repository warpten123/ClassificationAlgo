import math
from concurrent.futures import process
from hashlib import new
from multiprocessing.resource_sharer import stop
import docx2txt  # lib for reading docx files
import re
import pandas as pd
import numpy as np
import csv as csv
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import operator
# for data set
trainingSet = []
testSet = []
newDataset = []


# splits the data set to training set and testset
def dataSet(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvFile:
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

# second step in KNN algorithm
# hamming,Minkowski,Manhattan distance are options but euclidean is the most common
# euclidean formula = d = sqrt(pow(x-y)^2)


def getEuclideanDistance(pointOne, pointTwo, length):
    distance = 0
    for x in range(length):
        distance += pow((pointOne[x] - pointTwo[x]), 2)
    return math.sqrt(distance)


def manhattan_distance(instance1, instance2):
    absolute_differences = [abs(x1 - x2)
                            for (x1, x2) in zip(instance1, instance2)]
    return sum(absolute_differences)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = manhattan_distance(testInstance, trainingSet[x], length)
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse():
    print("getting the votes made by the neighbors")


def getAccuracy():
    print("accuracy of the model")


data1 = [2, 6, 3]
data2 = [5, 6, 7]
data = pd.read_csv("TFIDF.csv")
newCol = ['euclidean distance']
plt.scatter(data1, data2)
plt.show()
# print(data.values.tolist())
# result = manhattan_distance(data1, data2)

# print(result)

# distance = getEuclideanDistance(data['average'].tolist(), data['goal number'].tolist(), 3)
# ave = data['average'].tolist()
# goal_Num = data['goal number'].tolist()
# dist = []
# for x in range(len(data)):
#     distance = getEuclideanDistance(data['average'].tolist(), data['goal number'].tolist(), 3)
#     print(distance)

# dataSet(r'finalCSV.csv',0.66,trainingSet,testSet)
# print("Distance: " + str(distance))
# print('Training Set: ' + repr(len(trainingSet)))
# print('Test Set: ' + repr(len(testSet)))
