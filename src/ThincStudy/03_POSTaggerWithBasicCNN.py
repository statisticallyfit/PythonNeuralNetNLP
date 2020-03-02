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
    model.initialize(X = trainX[:5], Y = trainY[:5])

    for epoch in range(numIters):
        loss: float = 0.0
        # todo: type??
        batches = model.ops.multibatch(batchSize, trainX, trainY, shuffle=True)

        for X, Y in tqdm(batches, leave = False):
            YPred, backprop = model.begin_update(X = X)
            # todo type ??
            dLoss = []
            loss += ((YPred[epoch] - Y[epoch]) ** 2).sum()

        backprop(dLoss)
        model.finish_update(optimizer = optimizer)

        # todo type?
        score = evaluate(model, devX, devY, batchSize)
        #print(f"{i}\t{loss:.2f}\t{score:.3f}")
        print("Epoch: {} | Loss: {} | Score: {}".format(epoch, loss, score))


# todo types??
def evaluate(model: Model, devX: List[Array2d], devY: List[Array2d], batchSize: int) -> float:
    numCorrect: float = 0.0
    total: float = 0.0

    for X, Y in model.ops.multibatch(batchSize, devX, devY):
        # todo type of ypred??
        YPred = model.predict(X = X)

        for currYPred, currY in zip(YPred, Y):
            numCorrect += (currY.argmax(axis = 1) == currYPred.argmax(axis=1)).sum()

            # todo: what is the name of the dimension shape[0]?
            total += currY.shape[0]

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
