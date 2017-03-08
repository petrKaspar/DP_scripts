import numpy as np
from sklearn import svm

z = [2, 8, 5, 11]
print(z)

u = [[0.2], [0.8], [0.5], [0.11]]

print(u)
u=np.array(u)
print('u:',u)
# np.reshape(u, (-1, 2))

print(u.flatten())
t=u.flatten()
print(t)
b = t.reshape(-1)
print(b)

# Vzor pro vstup do SVM
u1 = [[0.2], [0.8], [0.5], [0.11]]
u2 = [[0], [15], [20], [21]]
u3 = [[1], [20], [25], [29]]
u4 = [[11], [10], [115], [219]]
u666 = [[0.9], [17], [22], [20]]

u1=np.array(u1)
u2=np.array(u2)
u3=np.array(u3)
u4=np.array(u4)
u666=np.array(u666)

q = []
q.append(u1.flatten())
q.append(u2.flatten())
q.append(u3.flatten())
q.append(u4.flatten())

print(u1)
print(u2)
print(u3)
u666 = u666.flatten()
print('u666: ',u666)
print(q)
print(np.array(q))

y = [0,1,1,1]
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(q,y)

# zobrazi se cisla skupiny, do ktere patri predikovane (predpovidane) hodnoty
print('u666 {} is in: '.format(u666), clf.predict(u666))
















