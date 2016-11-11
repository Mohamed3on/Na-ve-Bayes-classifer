import numpy
from sklearn import datasets

import functions

iris = datasets.load_iris()
X = iris.data[:, :]
Y = iris.target
full = numpy.c_[X, Y]
splitRatio = 0.67
train, test = functions.splitDataset(full, splitRatio)
separated = functions.separateByClass(train)

for k, x in separated.iteritems(): print x
