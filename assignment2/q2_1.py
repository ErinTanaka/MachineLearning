import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
import math
import operator

def findMax(arr):
    max=0
    for i in range(len(arr)):
        if arr[i]>max:
            max=arr[i]
    return max
def findMin(arr):
    min=arr[0]
    for i in range(len(arr)):
        if arr[i]<min:
            min=arr[i]
    return min

#FN scale0To1(minimum value in array x,
#               maximum value in array x,
#               array of values to scale)
def scale0To1(min, max, x):
    for i in range(len(x)):
        x[i]=(x[i]-min)/(max-min)
    return x

def normalizeData(arr):
    testNormArr=[]
    for i in range(len(arr[0])):
        itrArr=[row[i] for row in arr]
        myMax=findMax(itrArr)
        myMin=findMin(itrArr)
        if (myMax>0 or myMin<0):
            scaledArr=scale0To1(myMin, myMax, itrArr)
            scaledArr=np.array(scaledArr)
            testNormArr.append(scaledArr)
        else:
            itrArr=np.array(itrArr)
            testNormArr.append(itrArr)
    transposedArr=np.transpose(testNormArr)
    return transposedArr

def thresholdSort(set, featIndex):
    sorted=set[np.argsort(set[:, featIndex])]
    return sorted

def entropy(pos, neg):
    if (pos==0 or neg==0):
        return 0
    total=pos+neg
    p1=pos/total
    p2=neg/total
    math.log(p1, 2)
    math.log(p2, 2)
    Hs=-p1*math.log(p1, 2)-p2*math.log(p2, 2)
    return Hs

def calculateInformationGain(sPos, sNeg, lPos, lNeg, rPos, rNeg):
    sEntropy=entropy(sPos, sNeg)
    lEntropy=entropy(lPos, lNeg)
    rEntropy=entropy(rPos, rNeg)
    p1=(lPos+lNeg)/(sPos+sNeg)
    p2=(rPos+rNeg)/(sPos+sNeg)
    infoGain=sEntropy-(p1*lEntropy)-(p2*rEntropy)
    return infoGain

def getInfoGainForFeatN(data, featIndex):
    #index, info gain
    arr=[0,0]
    spos=0
    sneg=0
    for i in range(len(data)):
        if data[i][0] == 1:
            spos+=1
        else:
            sneg+=1
    thetaIndex=0
    isNextTheta=True
    while isNextTheta:
        for i in range(thetaIndex, len(sorted)):
            if i+1==len(data):
                isNextTheta=False
            if data[thetaIndex][0]!=data[i][0]:
                thetaIndex=i
                break
        #print("thetaIndex:"+ str(thetaIndex))

        theta=sorted[thetaIndex][featIndex]
        lpos=0
        lneg=0
        L=0
        while L<thetaIndex+1:
            if sorted[L][0]==1:
                lpos+=1
            else:
                lneg+=1
            L+=1
        rpos=0
        rneg=0
        R=thetaIndex+1
        while R<len(sorted):
            if sorted[R][0]==1:
                rpos+=1
            else:
                rneg+=1
            R+=1

        gain=calculateInformationGain(spos, sneg, lpos, lneg, rpos, rneg)
        if gain>arr[1]:
            arr[0]=thetaIndex
            arr[1]=gain
        # arr holds [thetaIndex of best info gain, and best info gain]
    return arr
def calcError(featIndex, theta, data):
    wrongCount=0
    for i in range(len(data)):
        # print(data[i][featIndex])
        # print(data[i][0])
        if data[i][featIndex]<=theta:
            if data[i][0]>0:
                wrongCount+=1
        else:
            #predict 1
            if data[i][0]<0:
                wrongCount+=1
    return wrongCount/len(data)

########################################################################
trainFile = str(sys.argv[1])
testFile = str(sys.argv[2])


trainX=[]
trainY=[]
with open(trainFile, mode = 'r') as trainFileCSV:
   trainCSVReader = csv.reader(trainFileCSV, delimiter = ',')
   numTrainLines = 0
   for row in trainCSVReader:
       #trainY.append(int(row.pop(0)))
       trainX.append(list(map(float, row)))
       numTrainLines+=1

testX=[]
testY=[]
with open(testFile, mode = 'r') as testFileCSV:
    testCSVReader = csv.reader(testFileCSV, delimiter = ',')
    numTestLines=0
    for row in testCSVReader:
        #testY.append(int(row.pop(0)))
        testX.append(list(map(float,row)))
        numTestLines+=1

# normTrainX=normalizeData(trainX)
# normTestX=normalizeData(testX)
normTestX=np.array(testX)
normTrainX=np.array(trainX)



bestInfoGains=[0,0,0]
for i in range(len(normTrainX[0])):
    featIndex=i+1
    if i+1==31:
        break
    sorted=thresholdSort(normTrainX, featIndex)
    # stuff holds [thetaIndex of best info gain, and best info gain]
    stuf=getInfoGainForFeatN(sorted, featIndex)
    if stuf[1]>bestInfoGains[1]:
        bestInfoGains=[featIndex, stuf[1], stuf[0]]


#print stump
resorted=thresholdSort(normTrainX, bestInfoGains[0])
thresh=resorted[bestInfoGains[2]][bestInfoGains[0]]
spos=0
sneg=0
for i in range(len(resorted)):
    if resorted[i][0] == 1:
        spos+=1
    else:
        sneg+=1
lpos=0
lneg=0
L=0
while L<bestInfoGains[2]+1:
    if resorted[L][0]==1:
        lpos+=1
    else:
        lneg+=1
    L+=1
rpos=0
rneg=0
R=bestInfoGains[2]+1
while R<len(resorted):
    if resorted[R][0]==1:
        rpos+=1
    else:
        rneg+=1
    R+=1
# print("Left: " + str(lpos)+ " " + str(lneg))
# print("Right: " + str(rpos) + " " + str(rneg))


print("         S: "+str(spos) +" (+)   |    " + str(sneg) + " (-)")
print("             /        X       \\         X: feat #" + str(bestInfoGains[0]) + " < " + str(thresh))
print("      p1 T  /                  \\ p2 F")
print("     "+str(lpos)+"(+) | "+str(lneg)+" (-)     "+str(rpos)+" (+) | "+str(rneg)+" (-)")


#print information gain
print("Information gain: "+ str(bestInfoGains[1]))

#training error
trainErr=calcError(bestInfoGains[0], thresh, normTrainX)
print("Training Error: " + str('{:.2%}'.format(trainErr)))

testingErr=calcError(bestInfoGains[0], bestInfoGains[1], normTestX)
print("Testing Error: " + str('{:.2%}'.format(testingErr)))
#testing error
