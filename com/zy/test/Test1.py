import numpy
from com.zy.chapter1.KNN import *
import os
import matplotlib
import matplotlib.pyplot as plt


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
path2 = path1 + "/data/datingTestSet2.txt"
datingDataMat, datingLabels = file2matrix(path2)
print(datingDataMat)
print(datingLabels)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels), 15.0*array(datingLabels), marker='x')
plt.xlabel('玩游戏所耗时间百分比')
plt.ylabel('每周消耗冰激凌公升数')
plt.legend('x')
plt.show()



# array = numpy.tile([1, 2], 5)
# print(array)
# array = numpy.tile([1, 2], (2, 3))
# print(array)
# array = numpy.tile([1, 2], (2, 3, 2))
# print(array)