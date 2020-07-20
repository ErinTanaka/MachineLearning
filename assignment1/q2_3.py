import csv, sys, scipy
import numpy as np
from scipy.stats import logistic
import matplotlib.pyplot as plt

trainingFile = str(sys.argv[1])
testingFile = str(sys.argv[2])
myLambda = float(sys.argv[3])
LEARNIN=0.0000000000000000005
trainX=[]
trainY=[]
xAxis=[]
with open(trainingFile, mode='r') as trainingFileCSV:
    trainingCSVReader = csv.reader(trainingFileCSV, delimiter=',')
    numTrainingLines = 0
    for row in trainingCSVReader:
        trainY.append(int(row.pop()))
        trainX.append(list(map(int, row)))
        numTrainingLines+=1
        #we gonna fix it good
testX=[]
testY=[]
with open(testingFile, mode='r') as testingFileCSV:
    testingCSVReader=csv.reader(testingFileCSV, delimiter=',')
    numTestingLines = 0
    for row in testingCSVReader:
        testY.append(int(row.pop()))
        testX.append(list(map(int, row)))
        numTestingLines+=1
print(testX)
w=np.zeros(256)
j=0
trainAccuracyArr=[]
testAccuracyArr=[]

gradient=np.zeros(256)
for i in range(numTrainingLines):
    print("i")
    x=trainX[i]
    #print("X: "+str(x))
    y=trainY[i]
    dotproduct=np.matmul(np.transpose(w).reshape(-1), x)
    denom=np.float(1+1/np.exp(dotproduct))
    yHat=np.float(1/denom)
    yHat=yHat+ 1/2 * myLambda * np.linalg.norm(w,2)
    #print("yhat"+str(yHat))
    if yHat>=0.5:
        yHat=1
    else:
        yHat=0
    error=y-yHat
    gradient=gradient + (np.outer(error, x))
    i+=0
w=w+(LEARNIN * gradient)
#training myLambda
secondWeight=np.zeros(256)
trainAccuracyArr=[]
while j<1000:
    xAxis.append(j+1)
    print("j"+str(j))
    gradient=np.zeros(256)
    errtrainct=0
    for k in range(numTrainingLines):
        x=trainX[k]
        y=trainY[k]
        dotproduct=np.matmul(np.transpose(secondWeight).reshape(-1), x)
        denom=np.float(1+1/np.exp(dotproduct))
        yHat=np.float(1/denom)
        yHat=yHat+ 1/2 * myLambda * np.linalg.norm(w,2)
        if yHat>=0.5:
            yHat=1
        else:
            yHat=0
        if(y-yHat==0):
            errtrainct+=1
        gradient=gradient+np.outer((y-yHat),x)
        k+=1
    secondWeight=secondWeight+(LEARNIN*gradient)
    trainAccuracyArr.append(errtrainct/numTrainingLines)
    j+=1


#testing accuracy
errTestCt=0
k=0
newxAxis=[]
for k in range(numTestingLines):
    print("k"+str(k))
    x=testX[k]
    y=testY[k]
    dotproduct=np.matmul(np.transpose(w).reshape(-1), x)
    denom=np.float(1+1/np.exp(dotproduct))
    yHat=np.float(1/denom)
    if yHat>=0.5:
        yHat=1
    else:
        yHat=0
    if(y-yHat==0):
        errTestCt+=1
    testAccuracyArr.append(errTestCt/numTestingLines)
    newxAxis.append(k+1)
    k+=1

testAccuracy=errTestCt/numTestingLines

print("trainAccuracy at 0 and end")
print(trainAccuracyArr[0])
print(trainAccuracyArr[len(trainAccuracyArr)-1])

print("testAccuracy at 0 and end")
print(testAccuracyArr[0])
print(testAccuracyArr[len(testAccuracyArr)-1])




#plot
plt.plot(xAxis, trainAccuracyArr)
plt.xlabel('Number of Gradient Iterations')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy Over ' + str(len(trainAccuracyArr)) + ' Iterations')
plt.show()
plt.savefig('trainingGraph.pdf',format='pdf')

plt.plot(newxAxis, testAccuracyArr)
plt.xlabel('Number of Gradient Iterations')
plt.ylabel('Testing Accuracy')
plt.title('Testing Accuracy Over ' + str(len(testAccuracyArr)) + ' Iterations')
plt.show()
plt.savefig('TestingGraph.pdf',format='pdf')
