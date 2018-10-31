from numpy import *
import os
import matplotlib.image as mpimg
from sklearn.svm import NuSVC
from skimage import feature as ft
import time

def loadImgFeature(rootdir):
    allDoc = os.listdir(rootdir)
    classLabelVector = []
    featureVector = []
    for k in range(len(allDoc)):
        doc = os.path.join(rootdir,allDoc[k])
        img = mpimg.imread(doc)
        features = ft.hog(img,  # input image
                          orientations=18,  # number of bins
                          pixels_per_cell=(2,2),  # pixel per cell
                          cells_per_block=(2,2),  # cells per blcok
                          block_norm='L1',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}, optional
                          transform_sqrt=True,  # power law compression (also known as gamma correction)
                          feature_vector=True,  # flatten the final vectors
                          visualize=False)  # return HOG map
        featureVector.extend(array(features).reshape(1,-1))
        filename = os.path.split(doc)[1]
        listFromLine = str(filename).strip().split('_')
        if listFromLine[0] == 'closed':
            classLabelVector.append(0)
        else:
            classLabelVector.append(1)
    return featureVector,classLabelVector

def main():
    start = time.time()
    trainX, trainY = loadImgFeature('trainingSet')
    testX, testY = loadImgFeature('testSet')
    clf = NuSVC()
    clf.fit(trainX,trainY)
    Z = clf.predict(testX)
    print("the total error rate is"+str(sum(Z!=testY) / float(len(testY))))

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
    print("\nthe total number of positive sample is %d,the positive sample error rate is %f." % (
    total1, error1 / float(total1)))
    print("\nthe total number of negative sample is %d,the negative sample error rate is %f." % (
    total0, error0 / float(total0)))
    print("spend time:%ss."%(time.time()-start))

if __name__ == '__main__':
    main()