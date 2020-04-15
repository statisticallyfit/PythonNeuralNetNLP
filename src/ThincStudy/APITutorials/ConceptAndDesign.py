# %% markdown
# Source: [https://thinc.ai/docs/concept#annotations:rRYs7HvtEeqjKcMi29YdQw](https://thinc.ai/docs/concept#annotations:rRYs7HvtEeqjKcMi29YdQw)
#
# # [Concept and Design](https://hyp.is/rRYs7HvtEeqjKcMi29YdQw/thinc.ai/docs/concept)
#
# ## [No (explicit) Computational Graph - Just Higher Order Functions](https://hyp.is/XMJRpk5FEeqjof-24zQhoA/thinc.ai/docs/concept)
#
#
# %% codecell
from thinc.types import *
from typing import *

def reduceSumLayer(X: Floats3d) -> Tuple[Floats2d, Callable[[Floats2d], Floats3d]]:
    Y: Floats2d = X.sum(axis = 1)

    # Backward pass runs from gradient-of-output (dY) to gradient-of-input (dX)
    # This means we will always have two matching pairs:
    # ---> (inputToForward, outputOfBackprop) == (X, dX), and
    # ---> (outputOfForward, inputOfBackprop) == (Y, dY) TODO ??
    def backpropReduceSum(dY: Floats2d) -> Floats3d:
        (dyFirstDim, dySecDim) = dY.shape
        dX: Floats3d = np.zeros(X.shape) # TODO thinc uses just `zeros` function -- from where??
        dX += dY.reshape((dyFirstDim, 1, dySecDim))

        return dX # d_inputs

    # outputs, backpropFunc
    return Y, backpropReduceSum


def reluLayer(inputs: Floats2d) -> Tuple[Floats2d, Callable[[Floats2d], Floats2d]]:
    mask: Floats2d = inputs >= 0
    outputs: Floats2d = inputs * mask

    def backpropRelu(dOutputs: Floats2d) -> Floats2d:
        return dOutputs * mask # == dInputs

    return outputs, backpropRelu

# %% markdown
# ## Example: Chain Combinator (using callbacks)
# The most basic we we will want to combine layers is in a feed-forward relationship. Calling this combinator `chain()`, after the calculus chain rule:

# %% codecell
def chain(firstLayer, secondLayer):
    def forwardChain(X):
        Y, getdX = firstLayer(X)
        Z, getdY = secondLayer(Y)

        def backpropChain(dZ):
            dY = getdY(dZ)
            dX = getdX(dY)

            return dX

        return Z, backpropChain

    return forwardChain

# %% markdown
# We can use the `chain()` combinator to build a function that runs our `reduceSUmLayer` and `reluLayer` layers in succession:
# %% codecell

# from thinc.api import glorot_uniform_init
import numpy as np

chainedForward = chain(firstLayer = reduceSumLayer, secondLayer = reluLayer)

B, S, W = 2, 10, 6 # (batch size, sequence length, width)

# TODO don't know which method thinc uses here: 'uniform' ???? Looked everywhere in thinc.api and thinc.backends but it's not available ...
X = np.random.uniform(low = 0, high = 1, size = (B, S, W))
dZ = np.random.uniform(low = 0, high = 1, size = (B, W))

# Returns Z, backpropChain
Z, getdX = chainedForward(X = X)
# The backprop chain in action:
dX = getdX(dZ = dZ)

assert dX.shape == X.shape

# %% markdown
# ## Example: Chain Combinator (No Callbacks)
# Our `chain` combinator works because our **layers return callbacks**, ensuring no distinction in API between the outermost layer and a layer that is part of a larger network. Imagine the alternative, where the function expects the gradient with respect to the output along its input:
# %% codecell
def reduceSum_noCallback(X: Floats3d, dY: Floats2d) -> Tuple[Floats2d, Floats3d]:
    Y: Floats2d = X.sum(axis = 1)

    # This was in the backprop method of reduceSumLayer():
    (dyFirstDim, dySecDim) = dY.shape

    dX: Floats3d = np.zeros(X.shape) # TODO thinc uses just `zeros` function -- from where??
    dX += dY.reshape((dyFirstDim, 1, dySecDim))

    return Y, dX


def relu_noCallback(inputs: Floats2d, dOutputs: Floats2d) -> Tuple[Floats2d, Floats2d]:
    mask: Floats2d = inputs >= 0
    outputs: Floats2d = inputs * mask

    # NOTE: this was in the backprop of the relu() method
    dInputs: Floats2d = dOutputs * mask
    #def backpropRelu(dOutputs: Floats2d) -> Floats2d:
    #    return dOutputs * mask
    #return inputs * mask, backpropRelu
    return outputs, dInputs


# How do we call `firstLayer`?
# We can't, because its signature expects dY  as part of its input – but we don't know dY yet!
# We can only  compute dY once we have Y. That's why layers must return callbacks.
def chain_noCallback(firstLayer, secondLayer):

    def forwardChain_noCallback(X, dZ):

        # NO CALLBACK:
        # Y, dX = firstLayer(X = X, dY = ???) # this is the stumbling block

        # WITH CALLBACK:  the callback way doesn't require firstLayer to take dY as its argument:
        # Y, getdX = firstLayer(X)

        raise NotImplementedError()

# %% markdown
# ### Key Design:
# The problem with no callbacks is more than just functional: the extra parameters passed in the functions in the above No Callback case are not just another kind of input variable to the network. The parameters are not part of the neural network design. We can't just say that parameters (like dY in the `reduceSum_noCallback`) are part of the network because that is not how we want to use the network. We [want the parameters of a layer to be an internal detail - **we don't want to have to pass in the parameters on each input**. ](https://hyp.is/HQ2zhE53EeqE19foO7IuTQ/thinc.ai/docs/concept)
#
# [Parameters must be handled differently from input variables (of a network) because we want to specify them at different times. We'd like to specify the parameters once *when we create the function* and then have them be an internal detail that doesn't affect the function's signature.](https://hyp.is/MgSHrk53EeqvG1tywXRcJg/thinc.ai/docs/concept)
#
# [The most direct approach is to introduce another layer of closures, and make the parameters and their gradients arguments to the outer layer. The gradients can then be incremented during the backward pass:](https://hyp.is/mz0GKnv9Eeq3Jg9d-UOOhA/thinc.ai/docs/concept)
# %% codecell

# TODO ERROR this code piece has errors: says dW, db are referenced before assignment!!!


def Linear(W, b, dW, db):

    def forwardLinear(X):  # X = inputs

        Y = X @ W.T + b  # Y = outputs

        def backwardLinear(dY):  # dY = d_outputs
            dW += dY.T @ X
            db += dY.sum(axis = 0)
            dX = dY @ W

            return dX    # dX = d_inputs

        return Y, backwardLinear

    return forwardLinear


(numBatches, nIn, nOut) = 128, 16, 32

# Initializing the inputs to neural network
W = np.random.uniform(low = 0, high = 1, size = (nOut, nIn)) # matrix
b = np.random.uniform(low = 0, high = 1, size = (nOut, )) # vector

# Initializing the derivatives
dW = np.zeros(shape = W.shape)
db = np.zeros(shape = b.shape)

X = np.random.uniform(low = 0, high = 1, size = (numBatches, nIn))
YTrue = np.random.uniform(low = 0, high = 1, size = (numBatches, nOut))

linear = Linear(W = W, b = b, dW = dW, db = db)
YOut, getdX = linear(X = X) # forward linear

dY = (YOut - YTrue) / numBatches # YTrue.shape[0]
dX = getdX(dY = dY) # backward linear

# Now we can do an optimization step like:
W -= 0.001 * dW
b -= 0.001 * db
dW.fill(0.0)
db.fill(0.0)

# %% markdown
# Use a [`Model` class to **keep track of parameters, gradients, dimensions**](https://hyp.is/si5GiE5NEeqQF1_W9FWCsQ/thinc.ai/docs/concept) since handling parameters and their gradients explicitly quickly gets unwieldy.
#
# Two possible approaches:
#
# 1. **Inheritance Approach:** introduce one class per layer type, with the forward pass implemented as a method on the class (like PyTorch)
# 2. **Composition Approach:**
#       * Each layer constructs a `Model` instance, and passes its `forward` function to this `Model` instance upon construction [(example is in the `thinc.layers.linear`)](https://hyp.is/-ceWpn7rEeqqmEdE6eVieg/thinc.ai/docs/api-layers). The `Model` object lets you pass in an [`init` function to support **shape inference**](https://hyp.is/bPmnGH7sEeqgGAMHdNcZBw/thinc.ai/docs/concept).
#       * In the `forward` method, the `Model` instance is passed in as a parameter, giving you access to the dimensions, parameters, gradients, attributes, and layers. THe second argument of `forward` is the input data and the third argument is a boolean that lets layers run differently during training and prediction (customary feature).
