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
    # ---> (outputOfForward, inputOfBackprop) == (dX, dY) TODO ??
    def backpropReduceSum(dY: Floats2d) -> Floats3d:
        (dyFirstDim, dySecDim) = dY.shape
        dX: Floats3d = np.zeros(X.shape) # TODO thinc uses just `zeros` function -- from where??
        dX += dY.reshape((dyFirstDim, 1, dySecDim))

        return dX

    return Y, backpropReduceSum

# %% codecell
def reluLayer(inputs: Floats2d) -> Tuple[Floats2d, Callable[[Floats2d], Floats2d]]:
    mask: Floats2d = inputs >= 0

    def backpropRelu(dOutputs: Floats2d) -> Floats2d:
        return dOutputs * mask

    return inputs * mask, backpropRelu

# %% markdown
# ## Example: Chain Combinator
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
# Our `chain` combinator works because our **layers return callbacks**, ensuring no distinction in API between the outermost layer and a layer that is part of a larger network. Imagine the alternative, where the function expects the gradient with respect to the output along its input:
# %% codecell
def reduceSum_noCallback(X, dY):
    
