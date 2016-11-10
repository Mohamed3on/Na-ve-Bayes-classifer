from sklearn import datasets
import numpy
import random
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

iris = datasets.load_iris()
X = iris.data[:, :]
Y = iris.target
full=numpy.c_[X, Y]
splitRatio = 0.67
train, test = splitDataset(full, splitRatio)
print('Split {0} rows into train with {1} and test with {2}').format(len(full), len(train), len(test))