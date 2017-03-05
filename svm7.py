import numpy as np
import cv2
import matplotlib.pyplot as plt

training_set = []
training_labels = []

for file in listing1:
    img = cv2.imread(path1 + file)
    res=cv2.resize(img,(250,250))
    gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    xarr=np.squeeze(np.array(gray_image).astype(np.float32))
    m,v=cv2.PCACompute(xarr)
    arr= np.array(v)
    flat_arr= arr.ravel()
    training_set.append(flat_arr)
    training_labels.append(1)

trainData = np.float32(training_set)
responses = np.float32(training_labels)
svm = cv2.SVM()
svm.train(trainData, responses, params=svm_params)
svm.save('svm_data.dat')