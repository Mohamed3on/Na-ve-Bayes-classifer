import math
import random


def mean(numbers):
    return sum(numbers) / float(len(numbers))


# standard deviation
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


# split the data randomly into training/testing sets according to a given ratio
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))  # get a random row
 # deletes the row from the dataset and appends it to the training set
 # the undeleted ones are the test sets
        trainSet.append(copy.pop(index)) 

    return [trainSet, copy]


# returns the dataset split according to class, by iterating over each row
# and checking its class
# then appending the matching list in the separated dictionary with it
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[
            -1] not in separated:  #negative indices mean counting from the left, so in this line it looks at the last item (class value)
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)  #add this row to its corresponding class
    return separated


# the zip(*dataset) function returns multiple arrays, one for each attribute.
# we delete the last array because it's for the class value
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(
        *dataset)]
 #for each attribute, save its mean and st. deviation in the summaries matrix
 # then delete the last summary (class)
    del summaries[-1]
    return summaries


# get the mean and st. deviation of each attribute for each class value
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(
            instances)  #this creates a matrix containing the summaries of attributes for each class
    return summaries


#this calculates the gaussian probability of a given point
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


#this calculates the probability that a given input belongs to each class, and returns them as an array (this is the classifier)
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


#this takes the probabilites and returns the class with the highest one
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, 0
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue

    return bestLabel


#this gets the percentage for a given row that it belongs to each class, by dividing its probability over the sum of all probabilites
def getProbs(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    chances = []
    sumOfProbs = 0
    for classValue, probability in probabilities.iteritems():
        sumOfProbs += probability

    for classValue, probability in probabilities.iteritems():
        prob = (probability / sumOfProbs) * 100
        chances.append(round(prob, 2))
    return chances


#this returns a class's name, instead of its number
def getClass(summaries, inputVector):
    bestLabel = predict(summaries, inputVector)

    if (bestLabel == 0.0):
        bestValue = "iris-setosa"
    elif (bestLabel == 1.0):
        bestValue = "Iris-versicolor"
    elif (bestLabel == 2.0):
        bestValue = "Iris-virginica"
    return bestValue


#gets predictions for the whole test set
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


#get the accuracy of your predictions
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0
