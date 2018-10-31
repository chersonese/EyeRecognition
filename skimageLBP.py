import os
import matplotlib.image as mpimg
from skimage.feature import local_binary_pattern
from numpy import *
from sklearn.svm import NuSVC
import cv2
import time
import numpy as np

def calHistogram(ImgLBPope,maskx=4,masky=4):
    h=24//maskx
    w=24//masky
    exHistograms = mat(zeros((shape(ImgLBPope)[0], 256 * w * h)))
    for q in range(shape(ImgLBPope)[0]):
        Img = ImgLBPope[q].reshape(24,24)
        Histogram = mat(zeros((h*w,256)))
        for i in range(h):
            for j in range(w):
                mask = zeros(shape(Img), uint8)
                mask[i * maskx: (i + 1) * maskx, j * masky:(j + 1) * masky] = 255
                hist = cv2.calcHist([array(Img, uint8)], [0], mask, [256], [0, 256])
                Histogram[(i + 1) * (j + 1) - 1,:] = mat(hist).flatten()
        exHistograms[q,:] = Histogram.flatten().reshape(1,256*h*w)
    return exHistograms

method=['default','ror','uniform','nri_uniform']
def loadImgFeaturesAndLabels(dir,p=8,r=2,x=8,y=8,m=0):
    allDoc = os.listdir(dir)
    features = []
    labelVector = []
    for i in range(len(allDoc)):
        doc = os.path.join(dir,allDoc[i])
        img = mpimg.imread(doc)
        img = np.sqrt(img/float(np.max(img)))
        feature = local_binary_pattern(img,p,r,method=method[m])
        features.extend(feature.reshape(1,-1))
        filename = allDoc[i].split('_')[0]
        if filename == 'closed':
            labelVector.append(0)
        else:
            labelVector.append(1)
    features = calHistogram(features,x,y)
    return features,labelVector

if __name__ == '__main__':
    start = time.time()
    trainPath = 'trainingSet'
    testPath = 'testSet'
    trainX,trainY = loadImgFeaturesAndLabels(trainPath)
    testX,testY = loadImgFeaturesAndLabels(testPath)
    clf = NuSVC()
    clf.fit(trainX,trainY)
    Z = clf.predict(testX)
    print('the total error rate:' + str(sum(Z != testY) / float(len(testY))))

    error0,error1,total0,total1 = 0,0,0,0
    for i in range(len(Z)):
        if testY[i] == 0:
            total0 += 1
            if Z[i] != 0:
                error0 += 1
        else:
            total1 += 1
            if Z[i] != 1:
                error1 += 1
    print("\nthe total number of positive sample is %d,the positive sample error rate is %f" % (
    total1, error1 / float(total1)))
    print("\nthe total number of negative sample is %d,the negative sample error rate is %f" % (
    total0, error0 / float(total0)))
    print("spend time:%ss."%(time.time()-start))