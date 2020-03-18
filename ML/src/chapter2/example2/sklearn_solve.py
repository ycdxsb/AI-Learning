import numpy as np
from sklearn import neighbors
import os


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


if __name__ == "__main__":
    trainingFileList = os.listdir("trainingDigits")
    m = len(trainingFileList)
    trainLabels = []
    k = 3
    trainingMat = np.zeros((m, 1024))
    for i in range(0, m):
        label = int(trainingFileList[i].split("_")[0])
        trainLabels.append(label)
        trainingMat[i] = img2vector("trainingDigits/%s" % trainingFileList[i])

    testFileList = os.listdir('testDigits')
    testLabels = []
    m = len(testFileList)
    testMat = np.zeros((m, 1024))
    for i in range(0, m):
        label = int(testFileList[i].split("_")[0])
        testLabels.append(label)
        testMat[i] = img2vector("testDigits/%s" % testFileList[i])

    clf = neighbors.KNeighborsClassifier(k, weights='uniform')
    clf.fit(trainingMat, trainLabels)
    pridects = clf.predict(testMat)
    errorCount = 0
    for i in range(m):
        if(pridects[i] != testLabels[i]):
            print("%d:%d" % (testLabels[i], pridects[i]))
        errorCount += pridects[i] != testLabels[i]
    print(errorCount)
