import numpy
from com.zy.chapter1.KNN import *

dataSet, labels = createDataSet()
testX = array([1.2, 1.0])
k = 3
outputLabel = kNNClassify(testX, dataSet, labels, k)
print("input:", testX, "output:" + outputLabel)
testX = array([0.1, 0.3])
outputLabel = kNNClassify(testX, dataSet, labels, 3)
print("Your input is:", testX, "and classified to class: ", outputLabel)



# array = numpy.tile([1, 2], 5)
# print(array)
# array = numpy.tile([1, 2], (2, 3))
# print(array)
# array = numpy.tile([1, 2], (2, 3, 2))
# print(array)