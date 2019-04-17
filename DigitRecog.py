import os
from numpy.linalg import svd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time
import random
from Processer import organizeImages
from math import e, log
from scipy import optimize





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

#returns a gradient row vector based on approx calculations through finding secant lines
def gradientChecking(rolledTheta,X,  Y, hiddenSize):
    dx = pow(10, -4) #this essentially is delta x
    rolledGrad = np.zeros(np.shape(rolledTheta))
    #finds gradients for the first however many rows
    for i in range(0, 25):
        #adding dx
        theta_plus = rolledTheta.tolist()
        theta_plus[i][0] += dx
        theta_plus = np.array(theta_plus)
        #subtracting dx
        theta_minus = rolledTheta.tolist()
        theta_minus[i][0] -= dx
        theta_minus = np.array(theta_minus)
        #appending the secant line slope
        rolledGrad[i] = costFunction(theta_plus, X, Y, hiddenSize) - costFunction(theta_minus, X, Y, hiddenSize)
        rolledGrad[i] /= 2 * dx

    return rolledGrad


#used for quickly testing certain things
def test():
   z = np.array([[0,0,0],[0,0,0]])
   c = np.array([[0,-1, -2], [0,1,2]])
   counter = 0
   for i in range(np.shape(c)[0]):
       if(maxIndex(c[i].tolist())) == maxIndex(z[i].tolist()):
           counter+= 1
   print(counter / np.shape(c)[0])



#used to find the number itself for predictions
def maxIndex(list):
    max = list[0]
    index = 0

    for i in range(0, len(list)):
            if(list[i] > max):
                max = list[i]
                index = i
    return index


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

    return a3

#costFunction for the neural network
def costFunction(rolledTheta, *args):
    X,Y,k = args
    #getting the theta
    theta1, theta2 = unroll(rolledTheta, np.shape(X)[1], k, np.shape(Y)[1])

    a3 = feedForward(X, theta1, theta2, Y)# predictions from the feedforward algo
    #finding loss and error through crossEntropy
    loss = crossEntropy(a3, Y)
    m = np.shape(X)[0] #total number of examples
    J = np.sum(np.sum(loss))/m
    return J

#gradients for the neural network
def findGradients(rolledTheta, *args):
    X,Y,k = args
    #unrolling thetas
    theta1, theta2, = unroll(rolledTheta, np.shape(X)[1], k, np.shape(Y)[1])
    # calculating the sums
    z2 = np.dot(X, theta1)  # 1187 * 30
    a2 = sigmoid(z2)  # 1187 * 30
    # adding bias weight
    biasUnits = np.ones((np.shape(a2)[0], 1))
    x2 = np.hstack((biasUnits, a2))  # 1187 * 31
    # calculating sums once again
    z3 = np.dot(x2, theta2)  # 1187 * 10
    a3 = sigmoid(z3)
    #calculating the propogation error for each layer
    delta3 = a3 - Y #1187 * 10
    #finding sigmoid'(z3) and making sure columns match up
    biasZ2 = np.hstack((np.ones((np.shape(z2)[0], 1)), z2))
    deltaZ2 = deltaSigmoid(biasZ2) #1187 x 31
    #calculating prop error for the theta1
    delta2 = np.multiply(np.dot(delta3, theta2.T), deltaZ2) #1187 * 31
    #computing gradients from the deltas
    m = np.shape(X)[0] #total examples of training data
    grad_1 = np.matmul(X.T, delta2)/m #901 x 31
    grad_2 = np.matmul(x2.T, delta3)/m # 31 X 10

    return roll(grad_1[:, 1:], grad_2).flatten()

#method of calculating cost function
def crossEntropy(H, Y):
    return -1 * ( np.multiply(Y, np.log(H)) + np.multiply(1 -Y, np.log(1 - H)))

#used as a activation function
def sigmoid(X):
    return 1 / (1 + np.power(e, -1 * X))

#derivative of sigmoid function
def deltaSigmoid(X):
    return sigmoid(X) * (1 - sigmoid(X))


#returns the size of total images
def sizeOfAllExamples(directory, subDirectories):
    os.chdir(directory)
    examples = 0
    for i in subDirectories:
        os.chdir(directory + str(i))
        examples += len(os.listdir()) - 1

    return examples


#returns the flattend version of the 2d matrix
def roll(theta1, theta2):
    n = np.shape(theta1)[0]
    k = np.shape(theta1)[1]
    o = np.shape(theta2)[1]
    #rolling the matrices into a row vector
    rolledVect = np.vstack((np.reshape(theta1, newshape=(n*k, 1)),np.reshape(theta2, newshape=((k+1) * o, 1))))
    return rolledVect

#returns 2 matrices based on the unrolled vector
def unroll(v, n, k, o):

    index = n*k
    # unrolling the vector
    m1 = v[0: index]
    m1 = np.reshape(m1, newshape=(n, k))
    m2 = v[index:]
    m2 = np.reshape(m2, newshape=((k + 1), o))

    return m1, m2

#As of 3/26/2019: I am using Open CV to process images
def main():

    organizeImages()
    n = 30 * 30 #dimensions of the pngs
    numberOfClasses = 10 # number of distinct classes we need to wacth out for
    learningRate = 0.01 #scale at which gradients will be applied
    sourceDirectory = "/Users/64000340/Desktop/Train/" #directory where all our data is
    subDirectory = [0,1,2,3,4,5,6,7,8,9] #names of the folders in the directory
    os.chdir("/Users/64000340/Desktop/Train") #directory for folder for processing images

    X, Y = processImages(n,numberOfClasses,sourceDirectory, subDirectory) #gives us our inputs & answers matricies
    m = np.shape(X)[0] #number of training examples
    n = np.shape(X)[1] #number of features
    k = 30 #number of neurons in hidden layer
    theta1 = np.random.randn(n, k)  * np.sqrt(1/(k)) #901 x k
    theta2 = np.random.randn(k + 1, numberOfClasses) * np.sqrt(1/(numberOfClasses))  #k+1 x 10
    #checking gradients
    rolledTheta = roll(theta1, theta2)
    bestTheta = trainFMINCG(rolledTheta, X, Y, k)
    b1, b2 = unroll(bestTheta, np.shape(X)[1], k, np.shape(Y)[1])
    a3 = feedForward(X, b1, b2, Y)
    counter = 0
    for i in range(np.shape(a3)[0]):
        print("Predicted", maxIndex(a3[i].tolist()), "Actual", maxIndex(Y[i].tolist()))
        if(maxIndex(a3[i].tolist()) == maxIndex(Y[i].tolist())):
            counter+= 1
    print("Accuracy :", (counter/np.shape(X)[0]) * 100)



#used to train the network by using backpropogation
def trainManually(X, theta1, theta2, Y, iterations, alpha, Reg = None):
    counter = 0
    #used for plotting J(Train)
    costs =[]
    epochs =[]
    #total number of cycles the network will go through
    while counter < iterations:
        #checking how many its getting right
        J = costFunction(X, theta1, theta2, Y)
        #appending the costs, counter, and number correct to our data lists
        costs.append(J)
        epochs.append(counter)
        #getting the gradients
        grad_1, grad_2 = findGradients(X, theta1, theta2, Y)
        theta1 = theta1 - alpha *grad_1
        theta2 = theta2 - alpha *grad_2
        counter+= 1

    #graphing the costs vs epochs curve
    plt.plot(epochs, costs)
    plt.show()

#used to train the network with the fmincg

def trainFMINCG(rolledTheta, X , Y, k,  Reg = None):
    res = optimize.fmin_cg(f=costFunction, x0=rolledTheta.flatten(), fprime=findGradients, args=(X, Y, k), maxiter=500)
    return res



start = time()
main()
print("Time elapsed : ", time() - start)