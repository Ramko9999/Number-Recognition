import os
from numpy.linalg import svd
import numpy as np
import cv2
import matplotlib.pylab as plt
from time import time
import random
from Processer import organizeImages
from math import pow, e



#this function will return the normalized version of the training data
def featureScale(train_mat, possibleRange, possibleMean):

    train_mat = train_mat - possibleMean
    train_mat = train_mat / possibleRange

    return train_mat


#this function will return a reduced dimension array of the original inputs
def dimReduce(train_mat, number_of_dims):

    if(number_of_dims > np.shape(train_mat)[1]):
        print("Number of dimensions cannot be greater than than original dimensions")
        return None

    #finding covariance matrix
    m = np.shape(train_mat)[0]
    covariance = np.dot(train_mat.T, train_mat) * 1/m # n x n

    #applying SVD on the covariance mat
    U, S, V = svd(covariance) # n x n

    #actually finding the new reduced inputs
    reducedFeatures = U[:, 0:number_of_dims] # n x k
    reducedInputs = np.dot(train_mat, reducedFeatures) # m x k

    return reducedInputs


#used for quickly testing certain things
def test():
    a = np.random.rand(10 , 6)
    print(a)
    print()
    print( 1 /( 1 + np.power(e, -1 * a)))


#used to process images
def processImages(n, k,  directory, subDirecs):

    X = np.ones((1, n))  # dummy creation of the inputs matrix
    Y = np.zeros((1, k))  # dummy creation of answers matrix
    counter = 0
    total = sizeOfAllExamples(directory, subDirecs)
    # getting each file in the directory
    print(sizeOfAllExamples(directory, subDirecs))
    while (counter < total):

        r = random.randint(0, 9)  # randomly ordering the training data
        os.chdir(directory + str(r))  # accessing the training folder for digit

        if (len(os.listdir()) < 2):
            continue

        # manipulating the Y matrix in the same order as X
        answers = [0] * k  # the len of the list represents the total classes in classification
        answers[r] = 1

        # appending the answers to the Y array
        answers = np.array(answers)
        Y = np.vstack((Y, answers))

        # getting & reshaping the first digit image read as np array
        sequence = cv2.imread(os.listdir()[1], 0)
        sequence = np.reshape(sequence, (1, 900))

        # adding it to our X array
        X = np.vstack((X, sequence))
        os.remove(os.listdir()[1])

        counter += 1

    X = X[1:, :]  # getting rid of the ones
    Y = Y[1:, :]  # m x 10
    X = featureScale(X, 255, 127.5)  # m x n

    # adding the bias inputs to the X array
    biasInputs = np.ones((np.shape(X)[0], 1))
    X = np.hstack((biasInputs, X))

    return X,Y


#feedforward algorithm in neural network
def feedForward(X, theta1, theta2, Y):
    #calculating the sums
    z2 = np.dot(X, theta1) # 1187 * 30
    a2 = sigmoid(z2) # 1187 * 30
    #adding bias weight
    biasUnits = np.ones((np.shape(a2)[0],1))
    x2 = np.hstack((biasUnits, a2)) # 1187 * 31
    #calculating sums once again
    z3 = np.dot(x2, theta2) # 1187 * 10
    a3 = sigmoid(z3)


    #used to find the number itself
    def maxIndex(list):
        max = list[0]
        index = 0

        for i in range(0, len(list)):
            if(list[i] > max):
                max = list[i]
                index = i
        return index

    #used to display predictions vs answers
    for example in range(0, len(Y)):

        print(maxIndex(list(Y[example, :])), " ", maxIndex(list(a3[example, : ])))


#used as a activation function
def sigmoid(X):
    return 1 /( 1 + np.power(e, -1 * X))


#returns the size of total images
def sizeOfAllExamples(directory, subDirectories):
    os.chdir(directory)
    examples = 0
    for i in subDirectories:
        os.chdir(directory + str(i))
        examples += len(os.listdir()) - 1

    return examples


#As of 3/26/2019: I am using Open CV to process images
def main():

    organizeImages()
    n = 30 * 30 #dimensions of the pngs
    numberOfClasses = 10 # number of distinct classes we need to wacth out for

    sourceDirectory = "/Users/64000340/Desktop/Train/" #directory where all our data is
    subDirectory = [0,1,2,3,4,5,6,7,8,9] #names of the folders in the directory
    os.chdir("/Users/64000340/Desktop/Train") #directory for folder for processing images

    X, Y = processImages(n,numberOfClasses,sourceDirectory, subDirectory) #gives us our inputs & answers matricies
    m = np.shape(X)[0] #number of training examples
    n = np.shape(X)[1] #number of features
    k = 30 #number of neurons in hidden layer
    theta1 = np.random.rand(n, k) - 0.5 #901 x 30
    theta2 = np.random.rand(k + 1, 10) - 0.5 #31 x 10

    feedForward(X, theta1, theta2, Y)



start = time()
main()

print(time() - start)