# %% markdown
# # Intro to Thinc's `Model` class, model definition, and methods
#
# Thinc uses a functional-programming approach to model definition, effective for:
# * complicated network architectures and,
# * use cases where different data types need to be passed through the network to reach specific subcomponents.
#
# This notebook shows how to compose Thinc models and use the `Model` class and its methods.
#
# *Principles:*
# * Thinc provides layers (functions to create `Model` instances)
# * Thinc tries to avoid inheritance, preferring function composition.

# %% codecell
import numpy
from thinc.api import Linear, zero_init

nI: int = 16
nO: int = 10
BATCH_SIZE: int = 128


nIn = numpy.zeros((BATCH_SIZE, nI), dtype="f")
nOut = numpy.zeros((BATCH_SIZE, nO), dtype="f")

# %% codecell
nIn.shape
nOut.shape

# %% codecell
model = Linear(nI = nI, nO = nO, init_W = zero_init)
model

model.get_dim("nI")
model.get_dim("nO")

print(f"Initialized model with input dimension nI={nI} and output dimension nO={nO}.")


# %% markdown
# *Key Point*: Models support dimension inference from data. You can defer some or all of the dimensions.
# %% codecell
modelDeferDims = Linear(init_W = zero_init)
modelDeferDims
print(f"Initialized model with no input/ouput dimensions.")

X = numpy.zeros((BATCH_SIZE, nI))
