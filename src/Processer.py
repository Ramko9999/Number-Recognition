import os
import cv2
import random
from time import time




#the main purpose of this program is to reorganize the training data and put it back in the test folder
def organizeImages(baseDirectory, targetDirectory):

    for i in range(0, 10): #assumes that the sub directory folders names are from 0-9 in the base & targetDirec
        os.chdir(baseDirectory+ str(i)) #test directory
        print(i)
        counter = 0

        for f in os.listdir():
            #reading the image & resizing
            if(f == ".DS_Store"):
                continue
            img = cv2.imread(f, 0)
            img = cv2.resize(img, (30,30))
            #writing to our test directory
            os.chdir(targetDirectory + str(i))
            cv2.imwrite(str(i) + " " + str(counter) + ".png", img) #creating a new gray scale image
            #heading back to base directory
            os.chdir(baseDirectory + str(i))
            counter += 1


#sending training data from a folder to the train folder: specific to user
def sendImage(baseDirectory, targetDirectory):

     counter = 0
     for i in range(0,10):
         os.chdir(baseDirectory + str(i))
         for file in os.listdir():
             img = cv2.imread(file, 1)
             os.chdir(targetDirectory + str(i))
             r = random.randint(0, 100000000)
             cv2.imwrite(str(i) + str(counter) + str(r) + ".png", img)
             os.chdir(baseDirectory + str(i))
             counter+= 1

#used for deleting files in a directory
def deleteFiles(baseDirectory, subDirecs):
    os.chdir(baseDirectory)
    for s in subDirecs:
        targetDirec = baseDirectory + s
        os.chdir(targetDirec)
        for f in os.listdir():
            os.remove(f)
'''
! THE CODE BELOW IS USED TO ORGANIZE ALL 15600 images I have into training & cv data, DON'T RANDOMLY RUN THIS
IT MIGHT MESS THINGS UP!

start = time()
nDirec_2 = "/Users/64000340/Desktop/Numbers2/"
nDirec_1 = "/Users/64000340/Desktop/Numbers/"
deleteFiles(nDirec_2, [str(x) for x in range(10)])
deleteFiles(nDirec_1, [str(x) for x in range(10)])
fileCounter = 0
base = "/Users/64000340/Desktop/All Data/"
digit_names = [str(x) for x in range(10)]
file_names = [str(x) for x in range(1, 27)]
test_nums = [156] * 10

for f in file_names:
    for d in digit_names:
        os.chdir("/Users/64000340/Desktop/All Data/" + f + "/" + d)
        for digit in os.listdir("/Users/64000340/Desktop/All Data/" + f + "/" + d):
            img = cv2.imread(digit, 0)
            img = cv2.resize(img, (30, 30))
            x = random.randint(0, 9)
            if (x == 0):
                if (test_nums[int(d)] == 0):
                    os.chdir(nDirec_1 + d)
                    cv2.imwrite(str(random.randint(0, 9999999)) + ".png", img)
                else:
                    os.chdir(nDirec_2 + d)
                    cv2.imwrite(str(random.randint(0, 9999999)) + ".png", img)
                    test_nums[int(d)] -= 1
            else:
                os.chdir(nDirec_1 + d)
                cv2.imwrite(str(random.randint(0, 9999999)) + ".png", img)
            os.chdir("/Users/64000340/Desktop/All Data/" + f + "/" + d)




print(time()-start, "seconds")


'''




