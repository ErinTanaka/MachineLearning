import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
import math
import operator

#FN: calcSimilarity
#input two lists of features to be compared
#output similarity of two entries calculated using euclidean distance
def calcSimilarity(trainFeats, testFeats):
    summ=0
    for i in range(len(trainFeats)):
        summ+=(trainFeats[i]-testFeats[i])**2
    similarity=math.sqrt(summ)
    return similarity

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

#returns 1d array of all distances between training set and single test instance
def getSimilarities(testInstance, trainingSet):
    allSimilarities=[]
    for i in range(len(trainingSet)):
        sim=calcSimilarity(trainingSet[i], testInstance)
        allSimilarities.append(sim)
    return allSimilarities

def getNeighbors(K, sims):
    neighbors=[]
    for i in range(K):
        neighbors.append(0)
        for j in range(len(sims)):
            if sims[j]<sims[(neighbors[i])]:
                neighbors[i]=j
        sims[neighbors[i]]=500
    return neighbors

def voteClassification(trainClass, neighborIndices, isLeaveOneOut, looClass):
    pos=0
    neg=0
    if(isLeaveOneOut):
        if looClass==1:
            pos=-1
        else:
            neg=-1
    for i in range(len(neighborIndices)):
        if trainClass[neighborIndices[i]]==1:
                pos+=1
        else:
            neg+=1
    if pos==neg:
        print("houston we have a problem")
    if pos>neg:
        return 1
    else:
        return -1
#normalized training and testing features
def KNNalg(K, trainX, trainClassifications, testX):
    testClassifications=[]
    for i in range(len(testX)):
        similarities=getSimilarities(testX[i], trainX)
        neighbors=getNeighbors(K, similarities)
        classified=voteClassification(trainClassifications, neighbors, False, -100)
        testClassifications.append(classified)
    return testClassifications
def KNNalgLoo(K, trainX, trainClassifications, testX, leaveOutInd):
    similarities=getSimilarities(testX, trainX)
    neighbors=getNeighbors(K+1, similarities)
    #print("leave one out neighbors:" + str(neighbors))
    classified=voteClassification(trainClassifications, neighbors, True, trainClassifications[leaveOutInd])
    return classified
##############MAIN##################

trainFile = str(sys.argv[1])
testFile = str(sys.argv[2])
K=int(sys.argv[3])

trainX=[]
trainY=[]
with open(trainFile, mode = 'r') as trainFileCSV:
   trainCSVReader = csv.reader(trainFileCSV, delimiter = ',')
   numTrainLines = 0
   for row in trainCSVReader:
       trainY.append(int(row.pop(0)))
       trainX.append(list(map(float, row)))
       numTrainLines+=1

testX=[]
testY=[]
with open(testFile, mode = 'r') as testFileCSV:
    testCSVReader = csv.reader(testFileCSV, delimiter = ',')
    numTestLines=0
    for row in testCSVReader:
        testY.append(int(row.pop(0)))
        testX.append(list(map(float,row)))
        numTestLines+=1

normTrainX=normalizeData(trainX)
normTestX=normalizeData(testX)

# classifications=KNNalg(K, normTrainX, trainY, normTestX)
# print(numTrainLines)
# print(numTrainLines/5)
i=1
xAxis=[]
trainingErrorArr=[]
testingErrorArr=[]
leaveOneOutVerArr=[]
#while needs to be <51 for turnin
while i<51:
    xAxis.append(i)
    #training error
    trainingErrClass=KNNalg(i, normTrainX, trainY, normTrainX)
    trainWrong=0
    j=0
    for j in range(len(trainingErrClass)):
        if trainingErrClass[j]!=trainY[j]:
            trainWrong+=1
    trainingError=trainWrong/len(trainingErrClass)
    trainingErrorArr.append(trainingError)

    #leave one out cross validation need to ask rasha
    looClassSum=0
    for j in range(len(normTrainX)):#j itr for which feature set to leave out
            #test featset
            testFeats=normTrainX[j]
            looClassJ=KNNalgLoo(i, normTrainX, trainY, testFeats, j)
            if looClassJ==trainY[i]:
                #correct classification
                looClassSum+=1
    LooVer=looClassSum/(len(normTrainX)-1)# not sure about the -1 but it kinda makes sense at the moment
    leaveOneOutVerArr.append(LooVer)
    #testing error
    testErrClass=KNNalg(i, normTrainX, trainY, normTestX)
    testWrong=0
    for j in range(len(testErrClass)):
        if testErrClass[j]!=testY[j]:
            testWrong+=1
    testingError=testWrong/len(testErrClass)
    testingErrorArr.append(testingError)

    i+=2
print("hells yeah")
#determiningK
ks=[]
for i in range(len(trainingErrorArr)):
    ks.append([trainingErrorArr[i], leaveOneOutVerArr[i], testingErrorArr[i]])
print(ks[11])
print(ks[12])
print(ks[13])
print(ks[14])
print(ks[15])
print("halleluya")
print(ks[len(ks)-3])
print(ks[len(ks)-2])
print(ks[len(ks)-1])




#plotting
plt.scatter(xAxis, trainingErrorArr)
plt.scatter(xAxis, leaveOneOutVerArr)
plt.scatter(xAxis, testingErrorArr)
plt.ylabel("Error in Decimal Form")
plt.xlabel("K-Nearest Neighbor")
plt.title('Testing Error, Leave-One-Out Verification, and Training Error Over ' + str(len(trainingErrorArr)) + ' Iterations')
plt.show()
plt.savefig('Hw2p1.pdf',format='pdf')
