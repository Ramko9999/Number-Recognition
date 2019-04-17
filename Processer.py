import os
import cv2




#the main purpose of this program is to reorganize the training data and put it back in the test folder
def organizeImages():

    baseDirectory = "/Users/64000340/Desktop/Numbers/"
    targetDirectory = "/Users/64000340/Desktop/Train/"


    for i in range(0, 10):
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
             cv2.imwrite(str(i) + str(counter) + ".png", img)
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


