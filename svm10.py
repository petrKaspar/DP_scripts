from datetime import time

import numpy as np
import itertools
import matplotlib.pyplot as plt
from networkx import reverse
from sklearn import svm, datasets
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import cv2
from os import listdir
import os
from os.path import isfile, join

imgDirectoryPath = 'images/lego/'
imgTrainingDirectory = 'images/lego/training3/'
imgTestDirectory = 'images/lego/test3/'
subDirectory = '/diff/'
filename = 'finalized_model.pkl'

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
    print(join(imgDirectoryPath, imgPath))
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    norm = 'withoutNorm'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        norm = 'Norm'
    else:
        print('Confusion matrix, without normalization')


    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('.'.__add__('\\saveFig' + norm + '.png'))#.format(time.strftime("%Y-%d-%m_%H:%M")))) # '.' pathToFigFolder


def getSVMinputData(trainingDir):
    X = np.array([[]])
    y = []
    index = 0
    for index, folder in enumerate(os.scandir(trainingDir)):
        folder = folder.name
        if (index == 0):
            nImages, X = (getMultiple_HOG(trainingDir + folder + subDirectory))
            y = y + [os.path.basename(folder)] * nImages
        else:
            nImages, x0 = (getMultiple_HOG(trainingDir + folder + subDirectory))
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
    clf = svm.SVC(kernel='linear', C=1, gamma='auto')#, C=1, gamma='auto'
    print('Training and Classification .........')
    clf.fit(X, y)

    training_classes = list(set(y))
    print(training_classes)
    # training_classes=['L-2x2_yellow', '4x2_red']


    # save the model to disk
    print('Saving the model to disk ........')
    joblib.dump(clf, filename)  #  compress=9   > parametr zmensi velikost asi 10x, ukladani je vsak pomale


    print('========================================================\nTest results:')
    X_test, y_test = getSVMinputData(imgTestDirectory)
    print(y_test)
    training_classes = list(set(y_test))
    # training_classes = training_classes[::-1]
    # print(training_classes)

    # training_classes = ['4x2_red', 'L-2x2_yellow',  'asdf']
    '''
    print('Loading a model data for testing from ', filename)
    clf = joblib.load(filename)
    '''
    predicted = clf.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, predicted)
    print("cnf_matrix: \n", cnf_matrix, '\n')


    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=training_classes,
                          title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=training_classes, normalize=True,
                          title='Normalized confusion matrix')


    # testImg = 'test4/4x4-Plate_green' + subDirectory + 'Img__201216_11_58_33-diff.png'
    # Q = getHOG(testImg)
    # print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))
    # testImg = 'test4/4x4-Plate_green' + subDirectory + 'Img__201216_11_58_36-diff.png'
    # Q = getHOG(testImg)
    # print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))
    # testImg = 'test4/4x4-Plate_green' + subDirectory + 'Img__201216_11_58_49-diff.png'
    # Q = getHOG(testImg)
    # print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))
    # testImg = 'test4/4x4-Plate_green' + subDirectory + 'Img__201216_11_58_55-diff.png'
    # Q = getHOG(testImg)
    # print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))
    # testImg = 'test4/4x4-Plate_green' + subDirectory + 'Img__201216_11_58_24-diff.png'
    # Q = getHOG(testImg)
    # print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))
    # testImg = 'test4/4x4-Plate_green' + subDirectory + 'Img__201216_11_58_29-diff.png'
    # Q = getHOG(testImg)
    # print('{} is in: '.format(testImg), clf.predict(Q.reshape(1, -1)))


    '''
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
    '''
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








