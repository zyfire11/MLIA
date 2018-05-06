from numpy import *
import operator
from os import listdir,path

# 生成数据集
def createDataSet():
    group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# knn分类
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]
    # print(numSamples)
    # print(tile(newInput, (numSamples, 1)))
    diff = tile(newInput, (numSamples, 1)) - dataSet
    # print("diff:", diff)
    squaredDiff = diff ** 2
    # print("squaredDiff:", squaredDiff)
    squaredDist = sum(squaredDiff, axis = 1)
    # print("squaredDist:", squaredDist)
    distance = squaredDist ** 0.5
    # print("distance:", distance)

    sortedDistIndices = argsort(distance)
    # print("sortedDistIndices:", sortedDistIndices)

    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    # return maxIndex
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 读取文件数据转矩阵
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

# 特征数据归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # print(m)
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 约会人分类测试
def datingClassTest(filePath):
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix(filePath)
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(hoRatio*m)
    errorCount = 0.0
    for i in range(numTestVecs):
        result = kNNClassify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print("the classifier came back with :%d, the real answer is : %d" % (result, datingLabels[i]))
        if(result != datingLabels[i]):
            errorCount+=1.0
    print("总的错误率：%f" % (errorCount/float(numTestVecs)))

# 约会数据输入分类识别
def classifyPerson(filePath):
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('玩游戏所占时间比是多少:'))
    ffMiles = float(input("每年做飞机的里程是多少:"))
    iceCream = float(input('每年吃多少升冰激凌:'))
    datingDataMat, datingLabels = file2matrix(filePath)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([percentTats, ffMiles, iceCream])
    result = kNNClassify(inArr, datingDataMat, datingLabels, 3)
    print("你可能是这种人：", resultList[result-1])

# 图像文件转向量（32*32=》1*1024）
def img2vector(filePath):
    returnVect = zeros((1, 1024))
    fr = open(filePath)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, i*32 + j] = int(lineStr[j])
    return returnVect

# 手写数字分类测试
def handwritingClassTest():
    trainFilePath = path.abspath("..") + "/data/digits/trainingDigits"
    testFilePath = path.abspath("..") + "/data/digits/testDigits"
    hwLabels = []

    trainingFileList = listdir(trainFilePath)
    m = len(trainingFileList)
    trainMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNum = int(fileNameStr.split("_")[0])
        hwLabels.append(classNum)
        trainMat[i, :] = img2vector(trainFilePath + "/" + fileNameStr)
    testFileList = listdir(testFilePath)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNum = int(fileNameStr.split('_')[0])
        testVector = img2vector(testFilePath + '/' + fileNameStr)
        result = kNNClassify(testVector, trainMat, hwLabels, 3)
        print('the classifier came back with %s, the real answer is %s' % (result, classNum))
        if(result != classNum):
            errorCount += 1.0
    print('错误率：%f' %(errorCount/float(m)))





