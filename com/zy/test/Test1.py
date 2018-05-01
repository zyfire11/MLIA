import numpy
from com.zy.chapter1.KNN import *
import os

dataSet, labels = createDataSet()
testX = array([1.2, 1.0])
k = 3
outputLabel = kNNClassify(testX, dataSet, labels, k)
print("input:", testX, "output:" + outputLabel)
testX = array([0.1, 0.3])
outputLabel = kNNClassify(testX, dataSet, labels, 3)
print("Your input is:", testX, "and classified to class: ", outputLabel)


# 约会
path1 = os.path.abspath('..')
print('父目录：' + path1)
path2 = path1 + "/data/datingTestSet.txt"
datingDataMat, datingLabels = file2matrix(path2)
datingDataMat.dtype
print('datingLabels:' + datingLabels)


# array = numpy.tile([1, 2], 5)
# print(array)
# array = numpy.tile([1, 2], (2, 3))
# print(array)
# array = numpy.tile([1, 2], (2, 3, 2))
# print(array)