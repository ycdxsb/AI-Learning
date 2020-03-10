import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
import operator
import os


def load_data(filename):
    f = open(filename)
    lines = f.readlines()
    dataMat = np.zeros((len(lines), 3))
    classLabel = []
    for i in range(len(lines)):
        line = lines[i].strip().split("\t")
        dataMat[i, :] = line[0:3]
        classLabel.append(int(line[-1]))
    return dataMat, classLabel


def show(dataMat, classLabel):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ["r", "g", "b"]
    for i in range(1, 4):
        x = []
        y = []
        for j in range(0, len(classLabel)):
            if classLabel[j] == i:
                x.append(dataMat[:, 0][j])
                y.append(dataMat[:, 1][j])
        ax.scatter(x, y, c=colors[i - 1])
    plt.show()


def norm_data(dataMat):
    minVals = dataMat.min(axis=0)
    maxVals = dataMat.max(axis=0)
    ranges = maxVals - minVals
    normddata = (dataMat - minVals) / ranges
    return normddata, ranges, minVals


def classify(x, dataMat, labels, k):
    dist = np.sum((x - dataMat) ** 2, axis=1) ** 0.5
    k_labels = [labels[index] for index in dist.argsort()[0 : k]]
    label = Counter(k_labels).most_common(1)[0][0]
    return label


if __name__ == "__main__":
    ratio = 0.2
    dataMat, classLabel = load_data("dataset.txt")
    dataMat, ranges, minVals = norm_data(dataMat)
    m = dataMat.shape[0]
    numTestVecs = int(m * ratio)
    errCount = 0
    for i in range(numTestVecs):
        result = classify(
            dataMat[i], dataMat[numTestVecs:m], classLabel[numTestVecs:m], 3
        )
        errCount += result != classLabel[i]
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr-minVals)/ranges,dataMat,classLabel, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])
    