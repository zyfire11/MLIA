import com.zy.chapter3.Trees as trees
from com.zy.chapter3.Trees import *

dataSet, labels = trees.createDataSet()
print(dataSet)
entropy = trees.calcShannonEnt(dataSet)
print(entropy)

# dataSet[0][2] = 'maybe'
# print(dataSet)

retDataSet = trees.splitDataSet(dataSet, 0, 1)
print(retDataSet)

bestFeture = trees.chooseBestFeatureToSplit(dataSet)
print(bestFeture)
myTree = trees.createTree(dataSet, labels)
print(myTree)