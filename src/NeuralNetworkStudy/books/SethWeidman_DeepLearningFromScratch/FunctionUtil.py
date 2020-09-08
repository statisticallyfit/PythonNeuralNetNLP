import numpy as np
import torch
import torch.tensor as Tensor

# Setting up the paths


import sys
import os

PATH: str = '/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP'

NEURALNET_PATH: str = PATH + '/src/NeuralNetworkStudy/books/SethWeidman_DeepLearningFromScratch'

os.chdir(NEURALNET_PATH)
assert os.getcwd() == NEURALNET_PATH

sys.path.append(PATH)
sys.path.append(NEURALNET_PATH)
assert PATH in sys.path
assert NEURALNET_PATH in sys.path

# Importing the types and derivative functions

from src.NeuralNetworkStudy.books.SethWeidman_DeepLearningFromScratch.TypeUtil import *
#from Chapter1_FoundationDerivatives import chainDerivTwo, chainTwoFunctions



def square(x: Tensor) -> Tensor:
     return np.power(x, 2)

def leakyRelu(x: Tensor) -> Tensor:
     '''Apply leaky relu function to each element in the tensor'''
     return np.maximum(0.2 * x, x)


def sigmoid(x: Tensor) -> Tensor:
     ''' Apply the sigmoid function to each element in the input array (Tensor)'''
     return 1 / (1 + np.exp(-x))



# My extra functions:
def cubic(x: Tensor) -> Tensor:
     return np.power(x, 3)

def quartic(x: Tensor) -> Tensor:
     return np.power(x, 4)

def quintic(x: Tensor) -> Tensor:
     return np.power(x, 5)

def sinT(x: Tensor) -> Tensor:
     return np.sin(x)

def cosT(x: Tensor) -> Tensor:
     return np.cos(x)
