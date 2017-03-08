import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import cv2
from os import listdir
from os.path import isfile, join

imgDirectoryPath = 'images/lego/'

def readMultipleImages():
    onlyfiles = [f for f in listdir(imgDirectoryPath) if isfile(join(imgDirectoryPath, f))]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread(join(imgDirectoryPath, onlyfiles[n]))

def getHOG(imgPath):
    hog = cv2.HOGDescriptor()
    im = cv2.cvtColor(cv2.imread(join(imgDirectoryPath, imgPath)), cv2.COLOR_BGR2GRAY)
    h = (hog.compute(im)).flatten()
    return np.array(h)

def getMultiple_HOG(folder):
    hog = cv2.HOGDescriptor()
    # h = [None]*10
    h=[]
    # h.append([1,2,3])
    # print(h)
    # h.append([4,5,6])
    # print(h)
    # print(h[:][1])
    # print(len(h[0]))

    # h = np.array([])

    onlyfiles = [f for f in listdir(imgDirectoryPath + folder) if isfile(join(imgDirectoryPath + folder, f))]
    images = np.empty(len(onlyfiles), dtype=object)
    print('List of imported images from {}:'.format(imgDirectoryPath + folder))
    for n in range(0, len(onlyfiles)):
        # images[n] = cv2.imread( join(imgDirectoryPath,onlyfiles[n]) )
        print(join(imgDirectoryPath + folder, onlyfiles[n]))
        im = cv2.cvtColor(cv2.imread(join(imgDirectoryPath + folder, onlyfiles[n])), cv2.COLOR_BGR2GRAY)
        h.append((hog.compute(im)).flatten())
        # print(hog.compute(im))
        # print('hog data: ', h)
    print(h)
    h = np.array(h)
    print(np.array(h))
    # h = np.array(h)
    # print(h.shape)
    # print(h)

    # y=[0,1]
    # clf = svm.SVC(kernel='linear', C=1, gamma='auto')
    # clf.fit(np.array(h), y)
    # clf = svm.SVC(kernel='linear', C=1.0)
    # clf.fit(X, y)
    # print('[0.58,0.76, 1] is in: ', clf.predict([0.58, 0.76, 1]))

    print('-------------------------------')
    # im = cv2.cvtColor(cv2.imread('images/img3.bmp'), cv2.COLOR_BGR2RGB)
    # plt.figure(0)
    # plt.imshow(im)
    # h = hog.compute(im)
    # print('hog data: ', h)
    # print('h.shape: ', len(h))
    # plt.scatter(h)
    print(h)
    return h

def trainingSMV(X, y):


    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0 # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=1,gamma='auto').fit(X, y)

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


if __name__ == "__main__":

    print('***************************************')
    X=(getMultiple_HOG('red/'))
    print('X = ',X)
    print('X.shape: ', len(X))

    print('+++++++++++++')
    x0 = (getMultiple_HOG('yellow/'))
    X = np.concatenate((X, x0))
    print('==================')
    print(X)
    print('X.shape: ', len(X))

    y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    clf = svm.SVC(kernel='linear', C=1, gamma='auto')
    clf.fit(X, y)


    print('===================================================\nTesting results:')
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








