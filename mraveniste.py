import os

trainingDirectory = 'images/lego/training/'

def getTrainingFoldersNames(trainingDirectory):
    trainingFolders = []
    for x in os.walk(trainingDirectory):
        trainingFolders.append(x[0])
    return trainingFolders

for trainingFolder in (getTrainingFoldersNames(trainingDirectory)):
    print(os.path.basename(trainingFolder))

for index, x in enumerate(os.scandir(trainingDirectory)):
    # folder = x[0]
    print(index, x.name)
    # print(os.path.basename(folder))
    # trainingFolders.append(x[0])

# print('--------------------')
# print(folders)
# print('*********************')
# print(folders[0])
# print(folders[1])
# print(folders[2])
s=""
log = [str(2), '/', str(20),' ']
print(s.join(log))

mylist = [u'nowplaying', u'PBS', u'PBS', u'nowplaying', u'job', u'debate', u'thenandnow']
myset = set(mylist)
print(myset)
print(list(myset))


