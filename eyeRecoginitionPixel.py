# coding=gbk
from numpy import *
import os
import matplotlib.image as mpimg
from sklearn.svm import NuSVC
import time

def getImgInf(rootdir):
    allDoc = os.listdir(rootdir)
    classLabelVector = []
    returnFinal = []
    for k in range(0,len(allDoc)):
        doc = os.path.join(rootdir,allDoc[k])
        img = mpimg.imread(doc)
        m,n = shape(img)
        returnMat = zeros((1,m*n))
        for i in range(m):
            for j in range(n):
                returnMat[0,n*i+j] = int(img[i,j])
        returnFinal.append(returnMat[0,:])
        filename = os.path.split(doc)[1]
        listFromLine = str(filename).strip().split('_')
        if listFromLine[0] == 'closed':
            classLabelVector.append(0)
        else:
            classLabelVector.append(1)
    return returnFinal,classLabelVector

if __name__ == '__main__':
    start = time.time()
    trainX,trainY = getImgInf('trainingSet')
    testX,testY = getImgInf('testSet')
    clf = NuSVC(kernel='linear')
    clf.fit(trainX, trainY)
    Z = clf.predict(testX)
    print("\nthe total error rate is: %f" % ( 1 - sum(Z==testY) / float(len(testX))))

    error0, error1, total0, total1 = 0, 0, 0, 0
    for i in range(len(Z)):
        if testY[i] == 0:
            total0 += 1
            if Z[i] != 0:
                error0 += 1
        else:
            total1 += 1
            if Z[i] != 1:
                error1 += 1
    print("\nthe total number of positive sample is %d,the positive sample error rate is %f"%(total1,error1/float(total1)))
    print("\nthe total number of negative sample is %d,the negative sample error rate is %f"%(total0,error0/float(total0)))
    print("spend time:%ss." % (time.time() - start))