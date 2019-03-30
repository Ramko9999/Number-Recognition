import os
from numpy.linalg import svd
import numpy as np
import cv2
import matplotlib.pylab as plt
from time import time
import random
from Organizer import organizeImages



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
    os.chdir("/Users/64000340/Desktop/Data")
    print(os.listdir())





#As of 3/26/2019: I am using Open CV to process images

def main():
    organizeImages()
    n = 30 * 30 #dimensions of the pngs
    os.chdir("/Users/64000340/Desktop/Train") #directory for folder for processing images
    X = np.ones((1,n)) #dummy creation of the inputs matrix
    Y = np.zeros((1,10)) #dummy creation of answers matrix

    #getting each file in the directory
    counter = 0
    while(counter < 600):
        r = random.randint(0,9) # randomly ordering the training data

        os.chdir("/Users/64000340/Desktop/Train/" + str(r)) # accessing the training folder for digit
        if(len(os.listdir()) < 2):
            continue
        #manipulating the Y matrix in the same order as X
        answers = [0,0,0,0,0,0,0,0,0,0]  #the len of the list represents the total classes in classification
        answers[r] = 1
        #appending the answers to the Y array
        answers = np.array(answers)
        Y = np.vstack((Y, answers))
        #getting & reshaping the first digit image read as np array
        sequence = cv2.imread(os.listdir()[1],0)
        sequence = np.reshape(sequence, (1,900))
        #adding it to our X array
        X = np.vstack((X, sequence))
        os.remove(os.listdir()[1])

        counter+= 1






    X = X[1:, :] #getting rid of the ones
    Y = Y[1: , :] # m x 10
    X = featureScale(X, 255, 127.5) # m x n

    #adding the bias inputs to the X array
    biasInputs = np.ones((np.shape(X)[0],1))
    X = np.hstack((biasInputs, X))

    print(X[:, 0])




start = time()
main()
print(time() - start)