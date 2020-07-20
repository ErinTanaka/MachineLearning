#In genereal I mostly want to confirm that we are taking all of our data from the correct places.
#I.E. in the ASE for the TEST data we are for sure using the Y array pulled from the test file but
#     do we need to calculate the corresponding yhat array based off of the testing data?
#Also are we doing our matrix operations in the proper order, or does it even matter? I keep doubting myself since
#     I still don't have a super solid grasp as to what is supposed to be happening


#handling cmd line args
import sys
import numpy as np
from numpy.linalg import inv
DEBUG = 0

trainingFileName = str(sys.argv[1])
testingFileName = str(sys.argv[2])

#get training contents and build training X matrix, and Y vector
fp=open(trainingFileName, 'r')
X = []
Y = []
tmpLine=fp.readline()
while tmpLine:
  one = [1]
  tmp=tmpLine.split()
  Y.append(float(tmp.pop()))
  tmpArr = one + tmp
  tmpArr = list(map(float, tmpArr))
  X.append(tmpArr)
  tmpLine=fp.readline()
fp.close()

if DEBUG:
  print("X: " + str(X))
  print("Y: "+ str(Y))

#get testing contents
fp=open(testingFileName, 'r')
XTest = []
YTest = []
line=fp.readline()
while line:
  one=[1]
  tmp=line.split()
  YTest.append(float(tmp.pop()))
  tmpArr=one+list(map(float,tmp))
  XTest.append(tmpArr)
  line=fp.readline()
fp.close()

#1.1
#compute weight vector
#w = xT*y (xT*x)^-1
np.array(X)
np.array(Y)

XT=np.transpose(X)
XTY = np.matmul(XT,Y)
XTX = np.matmul(XT,X)
XTXinv=inv(XTX)

w = np.matmul(XTY, XTXinv)
print("W: " + str(w))

#1.2
#compute ASE
#yhat vector y=w*(XT*X)*(XT)^-1
# y=w*XTX*(XT)^-1

#ASE for training data

# if code breaks around here use pinv instead
XTinv=np.linalg.pinv(XT)
#XTinv=inv(XT)
yHattmp=np.matmul(w, XTX)
yHat=np.matmul( XTinv, yHattmp)
if DEBUG:
  print("y hat: "+ str(yHat))
i = 0
tmpsum=0
while i<len(Y) :
  tmpsum+=(Y[i]-yHat[i])**2
  i+=1
aseTrain=tmpsum/(i+1)
print("ASE Training Data: " + str(aseTrain))

#ASE for Test Data
#need to double check these with fresh eyes there's an indexing error probly related to a bad matix computation
#TA question: does ase with test data use the same w as we calculated in part 1.1?
#TA question: does the test ASE need its own calculated yhat for computation?

#****this is sleep deprived brain barf and probly not using the corrrect things as of 11:59 am
#assuming I need a Yhat based off of the tesing data
# y=w*XTX*(XT)^-1
XTTest=np.transpose(XTest)
XTinvTest=np.linalg.pinv(XTTest)
XTXTest=np.matmul(XTTest,XTest)
yHatTest=np.matmul( XTXTest,w)
yHatTest=np.matmul( XTinvTest, yHatTest)

i = 0
tmpsum=0
while i<len(YTest):
    tmpsum+=(YTest[i]-yHatTest[i])**2
    i+=1
aseTest=tmpsum/(i+1)
print("ASE Testing Data: " + str(aseTest))
