# http://jmgomez.me/a-fruit-image-classifier-with-python-and-simplecv/

class Trainer():

    def __init__(self,classes, trainPaths):
        self.classes = classes
        self.trainPaths = trainPaths


    def getExtractors(self):
        hhfe = HueHistogramFeatureExtractor(10)
        ehfe = EdgeHistogramFeatureExtractor(10)
        haarfe = HaarLikeFeatureExtractor(fname='../SimpleCV/SimpleCV/Features/haar.txt')
        return [hhfe,ehfe,haarfe]

    def getClassifiers(self,extractors):
        svm = SVMClassifier(extractors)
        tree = TreeClassifier(extractors)
        bayes = NaiveBayesClassifier(extractors)
        knn = KNNClassifier(extractors)
        return [svm,tree,bayes,knn]

    def train(self):
        self.classifiers = self.getClassifiers(self.getExtractors())
        for classifier in self.classifiers:
            classifier.train(self.trainPaths,self.classes,verbose=False)

    def test(self,testPaths):
        for classifier in self.classifiers:
            print classifier.test(testPaths,self.classes,verbose=False)

   def visualizeResults(self,classifier,imgs):
        for img in imgs:
            className = classifier.classify(img)
            img.drawText(className,10,10,fontsize=60,color=Color.BLUE)
        imgs.show()


classes = ['pear', 'orange', 'strawberry', ]


def main():
    trainPaths = ['./post/' + c + '/train/' for c in classes]
    testPaths = ['./post/' + c + '/test/' for c in classes]

    trainer = Trainer(classes, trainPaths)
    trainer.train()
    tree = trainer.classifiers[1]

    imgs = ImageSet()
    for p in testPaths:
        imgs += ImageSet(p)
    random.shuffle(imgs)

    print
    "Result test"
    trainer.test(testPaths)

    trainer.visualizeResults(tree, imgs)


main()