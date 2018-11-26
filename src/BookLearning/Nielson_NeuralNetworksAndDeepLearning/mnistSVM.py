"""
mnist_svm
~~~~~~~~~
A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM (supervised vector machine)classifier."""

#### Libraries
# My libraries
import mnistLoader

# Third-party libraries
from sklearn import svm

def svmAlgo(): # not really the algo, using scikit learn
    trainingData, validationData, testData = mnistLoader.load_data()
    # train it
    classifier = svm.SVC()
    classifier.fit(trainingData[0], trainingData[1])
    # test
    predictions = [int(A) for A in classifier.predict(testData[0])]
    numCorrect = sum(int(A == y) for A, y in zip(predictions, testData[1]))

    print("Baseline classifier using an SVM (support vector machine)")
    print(str(numCorrect) + " of " + str(len(testData[1])) + " values correct.")


def main():
    svmAlgo()

if __name__ == "__main__":
    main()