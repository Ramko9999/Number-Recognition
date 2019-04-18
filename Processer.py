import os
import cv2
import random




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

data = ["0021_CH4M", "0018_CHXX", "0016_CH3F","0023_IT3M", "0022_AT3M", "0015_CH2M"]
for datum in data:
    sendImage("/Users/64000340/Desktop/" + datum +"/", "/Users/64000340/Desktop/Numbers/")