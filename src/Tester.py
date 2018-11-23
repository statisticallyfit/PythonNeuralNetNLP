
import sys
#sys.path.append('data/')
import mnistLoader
import network

#sys.path.append('Algo_StochasticGradientDescent')
#import network

#sys.path.append('Algo_SupportVectorMachineClassifier')
#import mnistSVM




def main():
    trainingData, validationData, testData = mnistLoader.load_data_wrapper()


    ### Chapter 1 --------------------------------------------------------------------------------

    # note: there are 28 x 28 = 784 input neurons because of 28 by 28 pixels in the image
    # note: there are 10 output neurons because in each output represents a number: 0...9
    net_30 = network.Network([784, 30, 10])
    # epochs = 30, minibatchsize = 10, learning rate eta = 3
    net_30.SGD(trainingData, 30, 10, 3, testData=testData) # classification rate keeps growing as the network learns


    net_100 = network.Network([784, 100, 10]) # network with 100 hidden layers
    # epochs = 30, minibatchsize = 10, learning rate eta = 3 are hyperparameters
    # note: epochs = number of minibatch training data sets.
    net_100.SGD(trainingData, 30, 10, 3, testData=testData)

    ### Chapter 1 --------------------------------------------------------------------------------



if __name__ == '__main__':
    main()




