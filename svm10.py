import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import cv2
from os import listdir
import os
from os.path import isfile, join

imgDirectoryPath = 'images/lego/'
imgTrainingDirectory = 'images/lego/training2/'
imgTestDirectory = 'images/lego/test2/'

def readMultipleImages():
    onlyfiles = [f for f in listdir(imgDirectoryPath) if isfile(join(imgDirectoryPath, f))]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread(join(imgDirectoryPath, onlyfiles[n]))

def getTrainingFoldersNames(trainingDirectory):
    trainingFolders = []
    for x in os.walk(trainingDirectory):
        trainingFolders.append(x[0])
    return trainingFolders

def getHOG(imgPath):
    hog = cv2.HOGDescriptor()
    im = cv2.cvtColor(cv2.imread(join(imgDirectoryPath, imgPath)), cv2.COLOR_BGR2GRAY)
    h = (hog.compute(im)).flatten()
    return np.array(h)

def getMultiple_HOG(folder):
    hog = cv2.HOGDescriptor()
    # h = [None]*10
    h=[]

    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    images = np.empty(len(onlyfiles), dtype=object)
    print('List of imported images from {}:'.format(folder))
    for n in range(0, len(onlyfiles)):
        log = [str(n+1), '/', str(len(onlyfiles)),' ']
        print(''.join(log), join(os.path.basename(folder), onlyfiles[n]))
        im = cv2.cvtColor(cv2.imread(join(folder, onlyfiles[n])), cv2.COLOR_BGR2GRAY)
        h.append((hog.compute(im)).flatten())

    h = np.array(h)


    # h = np.array(h)
    # print(h.shape)
    # print(h)

    # y=[0,1]
    # clf = svm.SVC(kernel='linear', C=1, gamma='auto')
    # clf.fit(np.array(h), y)
    # clf = svm.SVC(kernel='linear', C=1.0)
    # clf.fit(X, y)
    # print('[0.58,0.76, 1] is in: ', clf.predict([0.58, 0.76, 1]))
    nImages = h.shape[0]
    # print('Count of images = ', nImages)
    print('------------------------------------------')
    # im = cv2.cvtColor(cv2.imread('images/img3.bmp'), cv2.COLOR_BGR2RGB)
    # plt.figure(0)
    # plt.imshow(im)
    # h = hog.compute(im)
    # print('hog data: ', h)
    # print('h.shape: ', len(h))
    # plt.scatter(h)

    return nImages, h

def trainingSMV(X, y):


    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0 # SVM regularization parameter
    svc = svm.SVC(kernel='gbf', C=1.0, gamma='auto').fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))

    plt.figure(2)
    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC with linear kernel')
    plt.show()


def getSVMinputData(trainingDir):
    X = np.array([[]])
    y = []
    index = 0
    for index, folder in enumerate(os.scandir(trainingDir)):
        folder = folder.name
        if (index == 0):
            nImages, X = (getMultiple_HOG(trainingDir + folder))
            y = y + [os.path.basename(folder)] * nImages
        else:
            nImages, x0 = (getMultiple_HOG(trainingDir + folder))
            y = y + [os.path.basename(folder)] * nImages
            X = np.concatenate((X, x0))

    print('Loaded {} images from {} folders: '.format(len(X), index + 1))
    print('X.shape = ', X.shape)

    return X, y

if __name__ == "__main__":
    # **************************************************************
    # imgTrainingDirectory = 'images/lego/training2/'
    # X=np.array([[]])
    # y = []
    # index = 0
    # for index, folder in enumerate(os.scandir(imgTrainingDirectory)):
    #     folder = folder.name
    #     if (index == 0):
    #         nImages, X = (getMultiple_HOG(imgTrainingDirectory + folder))
    #         y = y + [os.path.basename(folder)] * nImages
    #     else:
    #         nImages, x0 = (getMultiple_HOG(imgTrainingDirectory + folder))
    #         print(folder)
    #         print(os.path.basename(folder))
    #         print(nImages)
    #         y = y + [os.path.basename(folder)] * nImages
    #         X = np.concatenate((X, x0))
    #     # X.append(x0)
    # print(y)
    # print('Loaded {} images from {} folders: '.format(len(X), index+1))
    # print('X.shape = ',X.shape)

    # **************************************************************

    ### /////////////////////////////////////////////////////////////////////
    # # y = []
    # nImages, X=(getMultiple_HOG('images/lego/training/1x2_grey'))
    # # print('X = ',X)
    # # print('X.shape: ', len(X))
    # y = ['red/']*nImages
    #
    # nImages, x0 = (getMultiple_HOG('images/lego/training/1x2_grey'))
    # y = y + ['yellow/'] * nImages
    # X = np.concatenate((X, x0))
    # print('==================')
    # print('Count of all images (X.shape): ', len(X))
    ### /////////////////////////////////////////////////////////////////////


    # y = [8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1]
    # print(y)

    X, y = getSVMinputData(imgTrainingDirectory)
    clf = svm.SVC(kernel='linear', C=1, gamma='auto')
    clf.fit(X, y)

    # save the model to disk
    print('Saving the model to disk ........')
    filename = 'finalized_model.pkl'
    joblib.dump(clf, filename)  #  compress=9   > parametr zmensi velikost asi 10x, ukladani je vsak pomale


    print('========================================================\nTest results:')
    X_test, y_test = getSVMinputData(imgTestDirectory)
    clf = joblib.load(filename)
    predicted = clf.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, predicted)
    print("cnf_matrix: \n", cnf_matrix, '\n')


    testImg = 'r.bmp'
    Q = getHOG(testImg)
    print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))
    testImg = 'r1.bmp'
    Q = getHOG(testImg)
    print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))
    testImg = 'r2.bmp'
    Q = getHOG(testImg)
    print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))
    testImg = 'r3.bmp'
    Q = getHOG(testImg)
    print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))

    testImg = 'y.bmp'
    Q = getHOG(testImg)
    print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))
    testImg = 'y1.bmp'
    Q = getHOG(testImg)
    print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))
    testImg = 'y2.bmp'
    Q = getHOG(testImg)
    print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))
    testImg = 'y3.bmp'
    Q = getHOG(testImg)
    print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))

    # ---------------------
    testImg = 'y1-test.bmp'
    Q = getHOG(testImg)
    print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))
    testImg = 'y2-test.bmp'
    Q = getHOG(testImg)
    print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))

    testImg = 'r1-test.bmp'
    Q = getHOG(testImg)
    print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))
    testImg = 'r2-test.bmp'
    Q = getHOG(testImg)
    print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))
    # X.extend(getHOG('yellow/'))
    # print('X.shape: ', len(X))
    # print(X)
    # y = (0,1)
    # print(y)
    #
    # trainingSMV(X, y)

    # print('X: ',X)
    # trainingSMV()



    plt.show()








