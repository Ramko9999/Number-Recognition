import os
import numpy as np
import cv2
from sklearn.datasets import load_digits
import matplotlib.pylab as plt


#As of 3/26/2019: I am using Open CV to process images

def main():
    m = 10000
    os.chdir("/Users/64000340/Desktop/Training Data") #directory for folder for processing images
    training = np.ones((1,m)) #dummy initlization

    #getting each file in the directory
    for file in os.listdir():
        print()
        if (file == ".DS_Store"): #skip the .DS_Store
            continue
        else:

            Img = cv2.imread(file, 0) #read the image as gray scale
            #make the array a row vector: 10000 x 1

            Sequence = np.reshape(Img, (10000,1))
            Sequence = Sequence.T # 1 x 10000

            #appending the data from each image to a training data array
            training =  np.vstack((training,Sequence))

    training = training[1:, :] #getting rid of the ones

    # appending the bias inputs to the training array
    rows = np.shape(training)[0]
    biasInputs = np.ones((rows, 1))
    training = np.hstack((biasInputs, training)) # 2 x 10001

    weights = np.zeros((np.shape(training)[1], 1)) #10001 x 1








#used for quickly testing certain things
def test():
    a = np.ones(2 , 2)




main()




