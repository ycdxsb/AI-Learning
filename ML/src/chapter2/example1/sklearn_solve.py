import numpy as np
from sklearn import neighbors

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

def norm_data(dataMat):
    minVals = dataMat.min(axis=0)
    maxVals = dataMat.max(axis=0)
    ranges = maxVals - minVals
    normddata = (dataMat - minVals) / ranges
    return normddata, ranges, minVals

if __name__ == "__main__":
    dataMat, classLabel = load_data("dataset.txt")
    dataMat, ranges, minVals = norm_data(dataMat)
    k = 3
    clf = neighbors.KNeighborsClassifier(k, weights='uniform')
    clf.fit(dataMat, classLabel)
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    inArr = np.array([[ffMiles, percentTats, iceCream]])
    classifierResult = clf.predict((inArr-minVals)/ranges)
    print("You will probably like this person: ", resultList[classifierResult[0] - 1])
