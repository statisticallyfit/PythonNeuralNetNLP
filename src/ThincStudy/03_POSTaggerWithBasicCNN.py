# %% markdown
# [Source](https://github.com/explosion/thinc/blob/master/examples/03_pos_tagger_basic_cnn.ipynb)

# # Basic CNN Part-of-Speech Tagger with Thinc
# We implement a basic CNN for pos-tagging (without external dependencies) in Thinc, and train the model on the Universal Dependencies [AnCora corpus](https://github.com/UniversalDependencies/UD_Spanish-AnCora).
#
# This tutorial shows three different workflows:

# 1. Composing the model in **code**
# 2. Composing the model using **only config file**
# 3. Composing the model in **code and configuring it via config** (recommended approach)

# %% codecell
from thinc.api import prefer_gpu

prefer_gpu()

# %% markdown
# Define the helper functions for loading data, and training and evaluating a given model.
# * NOTE: need to call `model.initialize` with a batch of input and output data to initialize model and infer missing shape dimensions.
# %% codecell
import ml_datasets
from tqdm.notebook import tqdm
from thinc.api import fix_random_seed, Model
from thinc.optimizers import Optimizer

from thinc.types import Array2d
from typing import Optional, List


fix_random_seed(0)

def trainModel(model: Model, optimizer: Optimizer, numIters: int, batchSize: int):

    (trainX, trainY), (devX, devY) = ml_datasets.ud_ancora_pos_tags()
    # Need to do shape inference:
    model.initialize(X = trainX[:5], Y = trainY[:5])

    for epoch in range(numIters):
        loss: float = 0.0
        # todo: type??
        batches = model.ops.multibatch(batchSize, trainX, trainY, shuffle=True)

        for X, Y in tqdm(batches, leave = False):
            Yh, backprop = model.begin_update(X = X)
            # todo type ??
            dLoss = []

            for i in range(len(Yh)):
                dLoss.append(Yh[i])
                loss += ((Yh[i] - Y[i]) ** 2).sum()

            backprop(dLoss)
            model.finish_update(optimizer = optimizer)

        # todo type?
        score = evaluate(model = model, devX = devX, devY = devY, batchSize = batchSize)
        #print(f"{i}\t{loss:.2f}\t{score:.3f}")
        print("Epoch: {} | Loss: {} | Score: {}".format(epoch, loss, score))




# todo types??
def evaluate(model: Model, devX, devY, batchSize: int) -> float:

    numCorrect: float = 0.0
    total: float = 0.0

    for X, Y in model.ops.multibatch(batchSize, devX, devY):
        # todo type of ypred??
        Yh = model.predict(X = X)

        for yh, y in zip(Yh, Y):
            numCorrect += (y.argmax(axis = 1) == yh.argmax(axis=1)).sum()

            # todo: what is the name of the dimension shape[0]?
            total += y.shape[0]

    return float(numCorrect / total)


# %% markdown
# ## 1. Composing the Model in Code
# Here's the model definition, using ...
# * `>>` operator for the `chain` combinator.
# * `strings2arrays` to transform a sequence of strings to a list of arrays
# * `with_array` transforms sequences (the passed sequences of arrays) into a contiguous two-dimensional array on the
# way into and out of the model it wraps.
#
# Final model signature: `Model[Sequence[str], Sequence[Array2d]]`
# %% codecell
from thinc.api import Model, chain, strings2arrays, with_array, HashEmbed, expand_window, Relu, Softmax, Adam, warmup_linear

width: int = 32
vectorWidth: int = 16
numClasses: int = 17
learnRate: float = 0.001
numIters: int = 10
batchSize: int = 128

with Model.define_operators(operators = {">>": chain}):

    modelFromCode = strings2arrays() >> with_array(

        layer = HashEmbed(nO = width, nV = vectorWidth, column=0)
        >> expand_window(window_size=1)
        >> Relu(nO = width, nI = width * 3)
        >> Relu(nO = width, nI = width)
        >> Softmax(nO = numClasses, nI = width)
    )
# %% codecell
optimizer = Adam(learn_rate = learnRate)
# %% codecell
modelFromCode
# %% markdown
# Training the model now:
# %% codecell
trainModel(model = modelFromCode,
           optimizer = optimizer,
           numIters = numIters,
           batchSize = batchSize)
#


# %% markdown
# ## 2. Composing the Model via a Config File
# Thinc's config system lets describe **arbitrary trees of objects**:
#
# 1. The config can include values like hyperparameters or training settings, or references to functions and the values of their arguments.
# 2. Thinc then creates the config **bottom-up** so you can define one function with its arguments, then pass the return value into another function.
#
# To rebuild the model in the above config file we need to break down its structure:
#
# * `chain` (takes any number of positional arguments)
# * `strings2array` (with no arguments)
# * `with_array` (one argument **layer**)
#   * **layer:** `chain` (any number of positional arguments)
#   * `HashEmbed`
#   * `Relu`
#   * `Relu`
#   * `Softmax`
#
# `chain` takes arbitrarily many positional arguments (layers to compose). 
