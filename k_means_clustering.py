import math

def euclideanDistance (pointOne, pointTwo, length):
    distance = 0
    for x in range(length):
        distance += pow((pointOne[x] - pointTwo[x]),2)
    return math.sqrt(distance)

data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = euclideanDistance(data1, data2, 3)
print ('Distance: ' + repr(distance))