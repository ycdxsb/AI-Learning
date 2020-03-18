from numpy import *
import operator
import os
from collections import Counter


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def classify(x, dataMat, labels, k):
    dist = sum((x - dataMat) ** 2, axis=1) ** 0.5
    k_labels = [labels[index] for index in dist.argsort()[0: k]]
    label = Counter(k_labels).most_common(1)[0][0]
    return label


if __name__ == "__main__":
    trainingFileList = os.listdir("trainingDigits")
    m = len(trainingFileList)
    trainLabels = []
    k = 3
    trainingMat = zeros((m, 1024))
    for i in range(0, m):
        label = int(trainingFileList[i].split("_")[0])
        trainLabels.append(label)
        trainingMat[i] = img2vector("trainingDigits/%s" % trainingFileList[i])

    testFileList = os.listdir('testDigits')
    m = len(testFileList)
    errorCount = 0
    testMat = zeros((m, 1024))
    for i in range(0, m):
        label = int(testFileList[i].split("_")[0])
        vector = img2vector("testDigits/%s" % testFileList[i])
        classifierResult = classify(vector, trainingMat, trainLabels, k)
        if(classifierResult != label):
            print("%d:%d" % (label, classifierResult))
        errorCount += classifierResult != label
    print(errorCount)
