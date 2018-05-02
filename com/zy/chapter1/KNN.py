from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]
    print(numSamples)
    print(tile(newInput, (numSamples, 1)))
    diff = tile(newInput, (numSamples, 1)) - dataSet
    print("diff:", diff)
    squaredDiff = diff ** 2
    print("squaredDiff:", squaredDiff)
    squaredDist = sum(squaredDiff, axis = 1)
    print("squaredDist:", squaredDist)
    distance = squaredDist ** 0.5
    print("distance:", distance)

    sortedDistIndices = argsort(distance)
    print("sortedDistIndices:", sortedDistIndices)

    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex

def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[3]))
        index += 1
    return returnMat, classLabelVector


