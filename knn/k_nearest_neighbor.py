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


# second step in KNN algorithm
# hamming,Minkowski,Manhattan distance are options but euclidean is the most common
# euclidean formula = d = sqrt(pow(x-y)^2)

class KNN:
    def dataSet(self, filename, split, trainingSet=[], testSet=[]):
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

    def getEuclideanDistance(self, pointOne, pointTwo, length):
        distance = 0
        for x in range(length):
            distance += pow((float(pointOne[x]) - float(pointTwo[x])), 2)
        return math.sqrt(distance)

    def manhattan_distance(instance1, instance2):
        absolute_differences = [abs(x1 - x2)
                                for (x1, x2) in zip(instance1, instance2)]
        return sum(absolute_differences)

    def getKNeighbors(self, trainingSet, testInstance, k):
        distances = []
        length = len(testInstance)-1
        for x in range(len(trainingSet)):
            dist = self.getEuclideanDistance(
                testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def getResponse(self, neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(),
                             key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

    def getAccuracy(self, testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] is predictions[x]:
                correct += 1
        return (correct/float(len(testSet))) * 100.0

    def main(self):  # prepare data
        trainingSet = []
        testSet = []
        split = 0.67
        self.dataSet('iris.data', split, trainingSet, testSet)
        # generate predictions
        predictions = []
        k = 3
        for x in range(len(testSet)):
            neighbors = self.getKNeighbors(trainingSet, testSet[x], k)
            result = self.getResponse(neighbors)
            predictions.append(result)
            print('> predicted=' + repr(result) +
                  ', actual=' + repr(testSet[x][-1]))
        accuracy = self.getAccuracy(testSet, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')
        return accuracy


# trainingSet = []
# testSet = []
# dataSet(r'iris.data', 0.66, trainingSet, testSet)``
# print('Train: ' + repr(len(trainingSet)))
# print('Test: ' + repr(len(testSet)))
# trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
# testInstance = [5, 5, 5]
# k = 1
# neighbors = getKNeighbors(trainSet, testInstance, 1)
# print(neighbors)
