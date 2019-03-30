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
    covariance = np.dot(train_mat.T, train_mat) * 1/m # m x m

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
    X = np.ones((1,n)) #dummy initlization of the inputs matrix
    Y = np.zeros((1,10)) #dummy initilzation of answers matrix
    #getting each file in the directory

    counter = 0
    while(counter < 600):
        r = random.randint(0,9)
        os.chdir("/Users/64000340/Desktop/Train/" + str(r))
        if(len(os.listdir()) < 2):
            continue

        sequence = cv2.imread(os.listdir()[1],0)
        print(np.shape(sequence))
        sequence = np.reshape(sequence, (1,900))
        X = np.vstack((X, sequence))
        os.remove(os.listdir()[1])

        counter+= 1
        print(counter)





    training = X[1:, :] #getting rid of the ones

    training = featureScale(training, 255, 127.5) # m x n

    print(np.shape(training))




start = time()
main()
print(time() - start)