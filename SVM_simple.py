import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]



plt.figure(0)
plt.scatter(x,y)

X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

y = [0,1,0,1,0,1]

clf = svm.SVC(kernel='linear', C = 1.0)

clf.fit(X,y)

print(clf.predict([0.58,0.76]))

print(clf.predict([10.58,10.76]))

w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()



# from sklearn import metrics
# from sklearn.svm import SVC
# # fit a SVM model to the data
# X = np.array([[1, 2],
#               [5, 8],
#               [1.5, 1.8],
#               [8, 8],
#               [1, 0.6],
#               [9, 11]])
# y = [0,1,0,1,0,1]
#
# model = SVC()
# model.fit(X, y)
#
# print(model)
# # make predictions
# expected = y
# predicted = model.predict(X)
# # summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))
#
