import numpy
from sklearn import datasets
from sklearn.metrics import confusion_matrix

import functions


def loadDatasets():
    iris = datasets.load_iris()
    X = iris.data[:, :]
    Y = iris.target
    full = numpy.c_[X, Y]  # concatenate
    splitRatio = 0.67  # third of the data set for test, 2 thirds for training
    train, test = functions.splitDataset(full, splitRatio)
    return train, test


def printConfusion(confusion, accuracy):
    print "Confusion matrix: (0: Iris-setosa 1:Iris-versicolor 2:Iris-virginica"
    row_labels = ['Actual 0', 'Actual 1', 'Actual 2']
    print("Predicted   0  1  2")
    for row_label, row in zip(row_labels, confusion):
        print '%s [%s]' % (row_label, ''.join('%03s' % i for i in row))  # format and add labels to the matrix
    print (" Accuracy: {0}% ").format(accuracy)


def getInput(summaries):
    while True:
        print "please enter the item you want to test (4 floating number values)"
        s = raw_input()
        numbers = map(float, s.split())
        print("input is: {0}").format(numbers)

        try:

            probs = functions.getProbs(summaries, numbers)
            print (
                "probabilities of each class: \n Iris-setosa: {0}% \n Iris-versicolor: {1}% \n Iris-virginica: {2}%").format(
                probs[0], probs[1], probs[2])
            print("Predicted class: {0}").format(functions.getClass(summaries, numbers))
        except ZeroDivisionError:
            print "this example is not valid, it doesn't belong to any class, please try a valid one"


# initializing the values
train, test = loadDatasets()
separated = functions.separateByClass(train)
summaries = functions.summarizeByClass(train)
predictions = functions.getPredictions(summaries, test)
accuracy = functions.getAccuracy(test, predictions)
testValues = [row[-1] for row in test]
confusion = confusion_matrix(testValues, predictions)
printConfusion(confusion, accuracy)
getInput(summaries)
