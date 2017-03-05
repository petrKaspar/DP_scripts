# http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
#Split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print("X_train = ",X_train)
print("X_test = ",X_test)
print("y_train = ",y_train)
print("y_test = ",y_test)
# Linear Kernel
svc_linear = svm.SVC(kernel='linear', C=1)
svc_linear.fit(X_train, y_train)
predicted= svc_linear.predict(X_test)
cnf_matrix = confusion_matrix(y_test, predicted)
print(cnf_matrix)

# # Output
#
# [[16  0  0]
#  [ 0 13  5]
# [ 0 4 7]]