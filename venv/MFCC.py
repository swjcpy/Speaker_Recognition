import numpy as np
import librosa
import os
import shutil

def mfcc(fileName, nMFCC):
    # all of the test records are sampled with rate 22050
    y, sr = librosa.load(fileName)
    mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=nMFCC)
    return mfcc

def initialFiles(originalTimitDir):
    listOfFolders = os.listdir(originalTimitDir)
    print(len(listOfFolders))
    count = 0
    for folder in listOfFolders:
        newDir = "wavFiles/"+ str(count)
        os.mkdir(newDir)
        listOfFiles = os.listdir(originalTimitDir + "/" + folder)
        fileCount = 0
        for file in listOfFiles:
            if os.path.splitext(file)[1] == ".wav":
                shutil.copyfile(originalTimitDir + "/" + folder+ "/" + file, newDir+ "/"+ str(fileCount) + ".wav")
                fileCount += 1
        count += 1

def initialTrainingAndTestingDataForCNN(nMFCC, trainNum, trainSetColumnNum):
    listOfTrainingData = []
    listOfTrainingLabel = []
    listOfTestingData = []
    listOfTestingLabel = []

    for countOfSpeaker in range(11):
        for i in range(trainNum):
            # training file number is a given parameter
            file = "wavFiles/" + str(countOfSpeaker)+ "/"+ str(i)+ ".wav"
            mfccSet = mfcc(file,nMFCC)
            row,column = mfccSet.shape
            for t in range(column - trainSetColumnNum):
                slide = mfccSet[:,t : t + trainSetColumnNum]
                listOfTrainingData.append(slide)
                listOfTrainingLabel.append(countOfSpeaker)

        for i in range(trainNum,10):
            file = "wavFiles/" + str(countOfSpeaker) + "/" + str(i) + ".wav"
            mfccSet = mfcc(file, nMFCC)
            row, column = mfccSet.shape
            for t in range(column - trainSetColumnNum):
                slide = mfccSet[ :, t: t + trainSetColumnNum ]
                listOfTestingData.append(slide)
                listOfTestingLabel.append(countOfSpeaker)


    np.save("trainingData.npy",listOfTrainingData)
    np.save("trainingLabel.npy",listOfTrainingLabel)
    np.save("testingData.npy", listOfTestingData)
    np.save("testingLabel.npy", listOfTestingLabel)


# os.mkdir("wavFiles")
# initialFiles("F:\武汉大学\物联网\学习\\2016-2017\毕业设计\\timit_test_1-3\dr1")
# mfcc("wavFiles/0/0.wav",13)
initialTrainingAndTestingDataForCNN(13,8,13)