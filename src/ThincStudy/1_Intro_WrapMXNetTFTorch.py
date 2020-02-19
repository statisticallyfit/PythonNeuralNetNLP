# %% markdown
# Source: https://github.com/explosion/thinc/blob/master/examples/00_intro_to_thinc.ipynb

# # Intro to Thinc: Defining Model and Config and Wrapping PyTorch, TensorFlow and MXNet
#
# ## Defining Model in Thinc

# %% codecell
# from thinc.api import prefer_gpu
# prefer_gpu()  # returns boolean indicating if GPU was activated

# %% markdown
# Declaring data below for the whole file:
# Using ml-datasets package in Thinc for some common datasets including MNIST:
# %% codecell
import ml_datasets

(trainX, trainY), (devX, devY) = ml_datasets.mnist()
print(f"Training size={len(trainX)}, dev size={len(devX)}")

# %% markdown
# ### Step 1: Define the Model
#
# Defining a model with two *Relu-activated hidden layers*, followed by a *softmax-activated output layer*. Also add *dropout* after the two hidden layers to help model generalize better.
#
# The `chain` combinator: acts like `Sequential` in PyTorch or Keras since it combines a list of layers together with a feed-forward relationship.

# %% codecell
from thinc.api import chain, Relu, Softmax

n_hidden = 32
dropout = 0.2

model = chain(Relu(nO=n_hidden, dropout=dropout),
              Relu(nO=n_hidden, dropout=dropout), Softmax())

# %% codecell
model

# %% markdown
# ### Step 2: Initialize the Model
#
# Call `Model.initialize` after creating the model and pass in a small batch of input data X and small batch of output data Y. Lets Thinc *infer the missing dimensions* (when we defined the model we didn't tell it the input size `nI` or the output size `nO`)
#
# When passing in the data, call `model.ops.asarray` to make sure the data is on the right device (transforms the arrays to `cupy` when running on GPU)
# %% codecell
# Making sure the data is on the right device
trainX = model.ops.asarray(trainX)
trainY = model.ops.asarray(trainY)
devX = model.ops.asarray(devX)
devY = model.ops.asarray(devY)

# Initializing model
model.initialize(X=trainX[:5], Y=trainY[:5])
nI = model.get_dim("nI")
nO = model.get_dim("nO")

print(
    f"Initialized model with input dimension nI = {nI} and output dimension nO = {nO}"
)

# %% markdown
# ### Step 3: Train the Model
#
# Create optimizer and make several passes over the data, randomly selecting paired batches of the inputs and labels each time.
#
# ** Key difference between Thinc and other ML libraries:** other libraries provide a single `.fit()` method to train a model all at once, but Thinc lets you *shuffle and batch your data*.
# %% codecell
from tqdm.notebook import tqdm

def trainModel(data, model, optimizer, numIter: int, batchSize: int):
    (trainX, trainY), (devX, devY) = data
    # todo why need indices?
    indices = model.ops.xp.arange(trainX.shape[0], dtype="i")

    for i in range(numIter):
        # multibatch(): minimatch one or more sequences of data and yield lists with one batch per sequence.
        batches = model.ops.multibatch(batchSize, trainX, trainY, shuffle=True)

        for X, Y in tqdm(batches, leave=False):
            # begin_update(): run the model over a batch of data, returning the output and a callback to complete the backward pass.
            # Returned: tuple (Y, finishedUpdated), where Y = batch of output data, and finishedUpdate = callback that takes the gradient with respect to the output and an optimizer function, and returns the gradient with respect to the input.
            Yh, backprop = model.begin_update(X=X)

            backprop(Yh - Y)

            # finish_update(): update parameters with current gradients. The optimizer is called with each parameter and gradient of the model.
            model.finish_update(optimizer=optimizer)

        # Evaluate and print progress
        numCorrect: int = 0
        total: int = 0

        for X, Y in model.ops.multibatch(batchSize, devX, devY):
            # predict(X: InT) -> OutT: calls the model's forward function with is_train=False, and returns only the output, instead of the (output, callback) tuple
            Yh = model.predict(X=X)
            numCorrect += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
            # todo?
            total += Yh.shape[0]

        score = numCorrect / total

        print(f" {i}: {float(score):.3f}")


# %% codecell
from thinc.api import Adam, fix_random_seed

fix_random_seed(0)
adamOptimizer = Adam(0.001)
BATCH_SIZE: int = 128
NUM_ITERATIONS: int = 10
print("Measuring performance across iterations: ")

trainModel(data=((trainX, trainY), (devX, devY)),
           model=model,
           optimizer=adamOptimizer,
           numIter=NUM_ITERATIONS,
           batchSize=BATCH_SIZE)

# %% markdown
# ## Another Way to Define Model: Operator Overloading
#
# * Thinc lets you *overload operators* and bind arbitrary functions to operators like +, *, and >> or @.
# * The `Model.define_operators` contextmanager takes a dictionary of operators mapped to functions (typically combinators like `chain`)
# * Operators in the dict are onl valid for the `with` block
# %% codecell
# Example of using the operators:
from thinc.api import Model, chain, Relu, Softmax

numHidden: int = 32
dropout: float = 0.2

with Model.define_operators({">>": chain}):
    model = Relu(nO=numHidden, dropout=dropout) >> Relu(
        nO=numHidden, dropout=dropout) >> Softmax()

# %% markdown
# NOTE: bunch of things here in source tutorial about config files ...

# %% markdown
# ## Wrapping TensorFlow, PyTorch, and MXNet models
#
# Can wrap the underlying model using Thinc interface to get type hints and use config system.
#
# ### 1. Wrapping TensorFlow Models
# Tensorflow's `Sequential` layer is equivalent to Thinc's `chain`.
# Defining here model with two Relu and dropout and softmax output.
#
# %% codecell
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from thinc.api import TensorFlowWrapper, Adam

width: int = 32
nO: int = 10
nI: int = 784
dropout: float = 0.2

tfModel = Sequential()
tfModel.add(Dense(width, activation="relu", input_shape=(nI, )))
tfModel.add(Dropout(dropout))
tfModel.add(Dense(width, activation="relu", input_shape=(nI, )))
tfModel.add(Dropout(dropout))
tfModel.add(Dense(nO, activation="softmax"))
tfModel
# %% markdown
# The wrapped tensorflow model:
# %% codecell
wrappedTFModel = TensorFlowWrapper(tensorflow_model=tfModel)
wrappedTFModel

# %% markdown
# Training the wrapped tensorflow model:
# %% codecell
data = ml_datasets.mnist()
#data
# %% codecell
adamOptimizer = Adam(learn_rate=0.001)
adamOptimizer
# %% codecell
# Providing batch of input data and batch of output data to do shape inference.
wrappedTFModel.initialize(X=trainX[:5], Y=trainY[:5])

# %% codecell
# Training the model
NUM_ITERATIONS = 10
BATCH_SIZE = 128

trainModel(data=data,
           model=wrappedTFModel,
           optimizer=adamOptimizer,
           numIter=NUM_ITERATIONS,
           batchSize=BATCH_SIZE)
