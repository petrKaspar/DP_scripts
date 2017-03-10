# https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import confusion_matrix

style.use("ggplot")
from sklearn import svm

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]
z = [2, 8, 1.8, 8, 0.6, 11]

plt.figure(0)
plt.scatter(x,y)

X = np.array([[1,2,2],
             [5,8,8],
             [1.5,1.8,1],
             [8,8,9],
             [1,0.6,0.7],
             [9,11,7]])

print(X.shape)

y = ['a','asd','a','asd','a','asd']

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X,y)

# zobrazi se cisla skupiny, do ktere patri predikovane (predpovidane) hodnoty
print('[0.58,0.76, 1] is in: ', clf.predict([0.58,0.76, 1]))
print('[10.58,10.76, 9] is in: ', clf.predict([10.58,10.76, 9]))


X_test = np.array([[1.2,2.2, 1.2],
             [5.3,8.1,7.2],
             [1.2,1.1, 0.5]])
y_test = ['a','asd','a']
print(X_test)
predicted = clf.predict(X_test)
cnf_matrix = confusion_matrix(y_test, predicted)
print("cnf_matrix:\n ", cnf_matrix)

# ------- Vykresleni grafu i s linearni primkou, oddelujici obe skupiny v grafu
# w = clf.coef_[0]
# print(w)
#
# a = -w[0] / w[1]
#
# xx = np.linspace(0,12)
# yy = a * xx - clf.intercept_[0] / w[1]
#
# h0 = plt.plot(xx, yy, 'k-', label="non weighted div")
#
# plt.figure(1)
# plt.scatter(X[:, 0], X[:, 1], c = y)
# plt.legend()
# plt.show()