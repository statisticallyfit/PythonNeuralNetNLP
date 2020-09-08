
# %% codecell
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import ndarray
#%matplotlib inline
from typing import *

import torch
import torch.tensor as Tensor



# %% codecell
import sys
import os

PATH: str = '/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP'

NEURALNET_PATH: str = PATH + '/src/NeuralNetworkStudy/books/SethWeidman_DeepLearningFromScratch'

os.chdir(NEURALNET_PATH)
assert os.getcwd() == NEURALNET_PATH

sys.path.append(PATH)
sys.path.append(NEURALNET_PATH)
assert NEURALNET_PATH in sys.path


#from FunctionUtil import *

from src.NeuralNetworkStudy.books.SethWeidman_DeepLearningFromScratch.FunctionUtil import *

from src.NeuralNetworkStudy.books.SethWeidman_DeepLearningFromScratch.TypeUtil import *

# %% markdown
# ### Derivative Function:
# $$
# \frac{df}{du}(a) = \lim_{\Delta \leftarrow 0} \frac{f(a + \Delta) - f(a - \Delta)}{2 \times \Delta}
# $$
# %% codecell

def deriv(func: Callable[[Tensor], Tensor],
     inputTensor: Tensor,
     delta: float = 0.001) -> Tensor:
     '''
     Evaluates the derivative of a function "func" at every element in the "inputTensor" array.
     '''
     return (func(inputTensor + delta) - func(inputTensor - delta)) / (2 * delta)

# %% markdown
# ### Nested (Composite) Functions:
# $$
# g(f(x)) = y
# $$
# %% codecell
# Define how data goes through a chain for a list of length 2:
def chainTwoFunctions(chain: Chain, x: Tensor) -> Tensor:
     '''Evaluates two functions in a row'''

     assert len(chain) == 2, "Length of input 'chain' should be 2"

     f: TensorFunction = chain[0]
     g: TensorFunction = chain[1]

     return g(f(x))



# %% markdown
# ### Chain Rule
# Leibniz notation:
# $$
# \frac {d} {dx} (g(f(x))) = \frac {dg} {df} \cdot \frac {df}{dx}
# $$
# Prime notation:
# $$
# (g(f(x)))' = g'(f(x)) \cdot f'(x)
# $$

# %% codecell
# Chain rule for two composed functions:
def chainDerivTwo(chain: Chain,  inputRange: Tensor) -> Tensor:
     '''Uses the chain rule to compute the derivative of two nested functions: (g(f(x)))' = g'(f(x)) * f'(x) '''

     assert len(chain) == 2, "This function requires 'Chain' objects of length 2"


     assert inputRange.ndim == 1, "Function requires a 1-dimensional Tensor as inputRange"

     # TensorFunction = Callable[[Tensor], Tensor]
     f: TensorFunction = chain[0]
     g: TensorFunction = chain[1]

     # f(x)
     fx: Tensor = f(inputRange) # TODO is this list of tensor or just tensor?

     # df / dx (or df / du)
     df_dx: Tensor = deriv(f, inputRange)

     # dg / du (f(x))  (or dg / df)
     dg_du: Tensor = deriv(g, f(inputRange))

     # Multiplying these quantities together at each point:
     return dg_du * df_dx




# %% markdown
# Plot the results to show the chain rule works:
# %% codecell

def plotChain_2_3(ax, chain: Chain, inputRange: Tensor) -> None:
     """
    Plots a chain function - a function made up of
    multiple consecutive ndarray -> ndarray mappings -
    Across the input_range

     Parameters
     ----------
     ax : type
         matplotlib subplot for fplotting
     chain : Chain

     inputRange : Tensor


     Returns
     -------
     None

     """
     assert inputRange.ndim == 1, "Function requires a 1-dimensional tensor as inputRange"

     if len(chain) == 2:
          outputRange: Tensor = chainTwoFunctions(chain = chain, x = inputRange)
     elif len(chain) == 3:
          outputRange: Tensor = chainThreeFunctions(chain = chain, x = inputRange)


     ax.plot(inputRange, outputRange)



def plotChainDeriv_2_3(ax, chain: Chain, inputRange: Tensor) -> None:
     """Uses the chain rule to plot the derivative of a function consisting of two or three nested functions.

     Parameters
     ----------
     ax : type
         matplotlib subplot for plotting.
     chain : Chain

     inputRange : Tensor
         Description of parameter `inputRange`.

     Returns
     -------
     Tensor
          Description of returned object.

     """
     if len(chain) == 2:
          outputRange: Tensor = chainDerivTwo(chain = chain, inputRange = inputRange)

     elif len(chain) == 3:
          outputRange: Tensor = chainDerivThree(chain = chain, inputRange = inputRange)

     ax.plot(inputRange, outputRange)






# %% codecell
PLOT_RANGE: Tensor = Tensor(np.arange(-3, 3, 0.01))

chainSquareSigmoid: Chain = [square, sigmoid]
chainSigmoidSquare: Chain = [sigmoid, square]

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 8)) # 2 rows, 1 column

# First chain (first nesting is sigmoid inner, square outer)
plotChain_2_3(ax = ax[0], chain = chainSquareSigmoid, inputRange = PLOT_RANGE)
plotChainDeriv_2_3(ax = ax[0], chain = chainSquareSigmoid, inputRange = PLOT_RANGE)

ax[0].legend(["$f(x)$", "$\\frac{df}{dx}$"])
ax[0].set_title("Function and derivative for\n$f(x) = sigmoid(square(x))$")

# Second chain (second nesting is square inner, sigmoid outer)
plotChain_2_3(ax = ax[1], chain = chainSigmoidSquare, inputRange = PLOT_RANGE)
plotChainDeriv_2_3(ax = ax[1], chain = chainSigmoidSquare, inputRange = PLOT_RANGE)
ax[1].legend(["$f(x)$", "$\\frac{df}{dx}$"])
ax[1].set_title("Function and derivative for\n$f(x) = square(sigmoid(x))$");





# %% markdown
# ### Chain Rule For Three Composed Functions:
# The function:
# $$
# y = h(g(f(x)))
# $$
#
# Leibniz notation of chain rule:
# $$
# \frac{d}{dx}(h(g(f(x)))) = \frac{dh}{d(g \circ f)} \cdot \frac {dg}{df} \cdot \frac {df}{dx}
# $$
#
# Prime notation of chain rule:
# $$
# (h(g(f(x))))' = h'(g(f(x))) \cdot g'(f(x)) \cdot f'(x)
# $$

# %% codecell
def chainThreeFunctions(chain: Chain, x: Tensor) -> Tensor:
     '''Evaluates three functions in a row (composition)'''
     assert len(chain) == 3, "Length of input 'chain' should be 3"

     f: TensorFunction = chain[0] # leaky relu
     g: TensorFunction = chain[1] # square
     h: TensorFunction = chain[2] # sigmoid

     return h(g(f(x)))



# %% markdown
# Creating functions to calculate compositions and chain rule for any-length chain:

# %% codecell
def chainFunctions(chain: Chain, x: Tensor) -> Tensor:
     '''Evaluates n functions in a row (composition'''

     # Applying the innermost function in the chain (first) to the tensor argument, and then the outermost (last)  functions act on the result.

     head, *tail = chain
     acc: Tensor = head(x)

     for i in range(0, len(tail)):
         tensorFunction: TensorFunction = tail[i]
         acc: Tensor = tensorFunction(acc)

     return acc




# %% codecell

def chainDerivThree(chain: Chain, inputRange: Tensor) -> Tensor:
     """Uses the chain rule to compute the derivative of three nested functions.

     Parameters
     ----------
     chain : Chain
         TODO
     inputRange : Tensor
          TODO

     """

     assert(len(chain) == 3), "This function requires `Chain` objects to have length 3 (means 3 nested functions)"

     f: TensorFunction = chain[0] # leaky relu
     g: TensorFunction = chain[1] # square
     h: TensorFunction = chain[2] # sigmoid


     ### Forward Pass part (computing forward quantities, direct function application)
     # f(x)
     f_x: Tensor = f(inputRange)

     # g(f(x))
     g_f_x: Tensor = g(f_x)


     ### Backward pass (computing derivatives using quantities that make up the derivative)
     # dh / d(g o f) or dh / du where u = g o f
     dh_dgf: Tensor = deriv(func = h, inputTensor = g_f_x)

     # dg / df (or dg / du where u = f)
     dg_df: Tensor = deriv(g, f_x)

     # df/dx
     df_dx: Tensor = deriv(f, inputRange)


     # Multiplying these quantities as specified by chain rule:
     # return df_dx * dg_df * dh_dgf # TODO what happens when reversing the order here?
     return dh_dgf * dg_df * df_dx # same thing when different order because these are 1-dim tensors.


# %% codecell
def chainAccumulate(chain: Chain, x: Tensor)-> List[Tensor]:

     head, *tail = chain
     acc: Tensor = head(x)

     accs: List[Tensor] = list()
     accs.append(acc)

     for i in range(0, len(tail)):
         tensorFunction: TensorFunction = tail[i]
         acc: Tensor = tensorFunction(acc)
         accs.append(acc)

     return accs

# %% codecell

from functools import reduce
from operator import mul


def forwardPass(chain: Chain, x:Tensor) -> List[Tensor]:
     '''Forward pass: function composition calculations while keeping the results stored.'''
     # Calculating function compositions, but not including the last function in the list.
     forwardNestings: List[Tensor] = chainAccumulate(chain[0: len(chain) - 1], x)

     # Add x on top so result is same length as chain, for backward pass's
     # convenience.
     return  [x] + forwardNestings


def backwardPass(chain: Chain, forwards: List[Tensor]) -> List[Tensor]:

     derivList: List[Tensor] = list()

     for i in list(reversed(range(0, len(chain)))):

          tensorFunc: TensorFunction = chain[i]
          forwardResult: Tensor = forwards[i]

          # Aply the chain rule
          dTensorFunc_dResult: Tensor = deriv(func = tensorFunc, inputTensor = forwardResult)

          derivList.append(dTensorFunc_dResult)

     return reduce(mul, derivList)

def chainDeriv(chain: Chain, x: Tensor) -> List[Tensor]:
     # Result of the forward function composition: where n = length(chain), and:
     # f_0 = chain[0]
     # ...
     # f_n-1 = chain[n-1]
     # ... the last element in the forward compositions list tensor is f_n-1( f_n-2(... f_2 (f_1 (f_0 (x)))...))
     forwardCompositions: List[Tensor] = forwardPass(chain, x)

     # Apply the chain rule: calculate derivatives and multiply them as per chain rule to get the result tensor.
     chainRuleResult: List[Tensor] = backwardPass(chain, forwardCompositions)

     return chainRuleResult

# %% markdown
# Testing the abstract functions chainFunction and chainDeriv:
# %% codecell
x: Tensor = Tensor(np.arange(-3, 8)); x
chain: List[TensorFunction] = [leakyRelu, sigmoid, square, cubic, quartic, quintic, sinT, cosT]

# %% codecell

assert torch.equal(chainDeriv(chain[0:3], x), chainDerivThree(chain[0:3], x))
assert torch.equal(chainFunctions(chain[0:3], x), chainThreeFunctions(chain[0:3], x))

# %% codecell
chainDeriv(chain[0:3], x)
# %% codecell
chainDerivThree(chain[0:3], x)



# %% markdown
# Showing that the nested 3 derivative function works:

# %% codecell

# %% markdown
# Plot the results to show the chain rule works:
# %% codecell

def plotChain(ax, chain: Chain, inputRange: Tensor) -> None:

     assert inputRange.ndim == 1, "Function requires a 1-dimensional tensor as inputRange"

     outputRange: Tensor = chainFunctions(chain = chain, x = inputRange)

     ax.plot(inputRange, outputRange)



def plotChainDeriv(ax, chain: Chain, inputRange: Tensor) -> None:
     assert inputRange.ndim == 1, "Function requires a 1-dimensional tensor as inputRange"

     outputRange: Tensor = chainDeriv(chain = chain, x = inputRange)

     ax.plot(inputRange, outputRange)


# %% codecell
PLOT_RANGE: Tensor = Tensor(np.arange(-3, 3, 0.01))

chainReluSquareSigmoid: Chain = [leakyRelu, square, sigmoid]
chainReluSigmoidSquare: Chain = [leakyRelu, sigmoid, square]

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 8)) # 2 rows, 1 column

# First chain (first nesting is sigmoid inner, square outer)
plotChain(ax = ax[0], chain = chainReluSquareSigmoid, inputRange = PLOT_RANGE)
plotChainDeriv(ax = ax[0], chain = chainReluSquareSigmoid, inputRange = PLOT_RANGE)

ax[0].legend(["$f(x)$", "$\\frac{df}{dx}$"])
ax[0].set_title("Function and derivative for\n$f(x) = sigmoid(square(leakyRrelu(x)))$")

# Second chain (second nesting is square inner, sigmoid outer)
plotChain(ax = ax[1], chain = chainReluSigmoidSquare, inputRange = PLOT_RANGE)
plotChainDeriv(ax = ax[1], chain = chainReluSigmoidSquare, inputRange = PLOT_RANGE)

ax[1].legend(["$f(x)$", "$\\frac{df}{dx}$"])
ax[1].set_title("Function and derivative for\n$f(x) = square(sigmoid(leakyRelu(x)))$");


# %% codecell
