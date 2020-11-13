# %% [markdown]
# # Chapter 1: Foundations (Basic Matrix Derivatives)
# 
# Chapter 1 of Seth Weidman's Deep Learning from Scratch - Building with Python from First Principles. 
# 
# **My Enrichment:** Studying basic matrix derivatives with `sympy` and `torch`, using my appendix from `MatrixCalculusStudy` folder as background knowledge. 

# %% codecell
import matplotlib.pyplot as plt
import matplotlib
# NOTE: must comment out this inline statement below when debugging cells in VSCode else error occurs. 
%matplotlib inline 


import numpy as np
from numpy import ndarray

from typing import *
import itertools
from functools import reduce

from sympy import Matrix, Symbol, derive_by_array, Lambda, Function, MatrixSymbol, Identity, Derivative, symbols, diff, HadamardProduct
from sympy.abc import x, i, j, a, b

# %%
import torch
import torch.tensor as tensor

# Types

Tensor = torch.Tensor 
LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor

# %% codecell
import sys
import os

PATH: str = '/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP'

UTIL_DISPLAY_PATH: str = PATH + "/src/utils/GeneralUtil/"

NEURALNET_PATH: str = PATH + '/src/NeuralNetworkStudy/books/SethWeidman_DeepLearningFromScratch'

#os.chdir(NEURALNET_PATH)
#assert os.getcwd() == NEURALNET_PATH

sys.path.append(PATH)
sys.path.append(UTIL_DISPLAY_PATH)
sys.path.append(NEURALNET_PATH)
#assert NEURALNET_PATH in sys.path

# %%
#from FunctionUtil import *

from src.NeuralNetworkStudy.books.SethWeidman_DeepLearningFromScratch.FunctionUtil import *

from src.NeuralNetworkStudy.books.SethWeidman_DeepLearningFromScratch.TypeUtil import *
#from src.NeuralNetworkStudy.books.SethWeidman_DeepLearningFromScratch.Chapter1_FoundationDerivatives import *

from src.NeuralNetworkStudy.books.SethWeidman_DeepLearningFromScratch.FunctionUtil import *


from src.utils.GeneralUtil import *
from src.MatrixCalculusStudy.MatrixDerivLib.symbols import Deriv
from src.MatrixCalculusStudy.MatrixDerivLib.diff import diffMatrix
from src.MatrixCalculusStudy.MatrixDerivLib.printingLatex import myLatexPrinter

# For displaying
from IPython.display import display, Math
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax', latex_printer= lambda e, **kw: myLatexPrinter.doprint(e))

# %% [markdown]
# ## Derivative Function:
# $$
# \frac{df}{du}(a) = \lim_{\Delta \leftarrow 0} \frac{f(a + \Delta) - f(a - \Delta)}{2 \times \Delta}
# $$
# %% codecell

def deriv(func: TensorFunction, #Callable[[Tensor], Tensor],
     inputTensor: Tensor,
     delta: float = 0.001) -> Tensor:
     '''
     Evaluates the derivative of a function "func_i" at every element in the "inputTensor" array.
     '''
     return (func(inputTensor + delta) - func(inputTensor - delta)) / (2 * delta)

# %% [markdown]
# ## Nested (Composite) Functions:
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



# %% [markdown]
# ## Chain Rule
# Leibniz notation:
# $$
# \frac {d} {dx} (g(f(x))) = \frac {dg} {df} \times \frac {df}{dx}
# $$
# Prime notation:
# $$
# (g(f(x)))' = g'(f(x)) \times f'(x)
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




# %% [markdown]
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
PLOT_RANGE: Tensor = tensor(np.arange(-3, 3, 0.01))

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





# %% [markdown]
# ### Chain Rule For Three Composed Functions:
# The function:
# $$
# y = h(g(f(x)))
# $$
#
# Leibniz notation of chain rule:
# $$
# \frac{d}{dx}(h(g(f(x)))) = \frac{dh}{d(g \circ f)} \times \frac {dg}{df} \times \frac {df}{dx}
# $$
#
# Prime notation of chain rule:
# $$
# (h(g(f(x))))' = h'(g(f(x))) \times g'(f(x)) \times f'(x)
# $$

# %% codecell
def chainThreeFunctions(chain: Chain, x: Tensor) -> Tensor:
     '''Evaluates three functions in a row (composition)'''
     assert len(chain) == 3, "Length of input 'chain' should be 3"

     f: TensorFunction = chain[0] # leaky relu
     g: TensorFunction = chain[1] # square
     h: TensorFunction = chain[2] # sigmoid

     return h(g(f(x)))


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






# %% [markdown]
# Creating functions to calculate compositions and chain rule for any-length chain:
#
# ### Chain Rule An Arbitrary Number of Composed Functions:
# The function:
# $$
# y = f_n(f_{n-1}(...f_2(f_1(f_0(x)))...))
# $$
#
# Leibniz notation of chain rule:
# $$
# \frac{d}{dx}( f_n(f_{n-1}(...f_2(f_1(f_0(x)))...)) ) = \frac{df_n}{d(f_{n-1} \circ f_{n-2} \circ ... \circ f_0)} \times \frac {df_{n-1}}{df_{n-2}} \times ... \times \frac {df_2}{df_1} \times \frac{df_1}{df_0} \times \frac{df_0}{dx}
# $$
#
# Prime notation of chain rule:
# $$
# (f_n(f_{n-1}(...f_2(f_1(f_0(x)))...)))' = f_n'(f_{n-1}(...f_1(f_0(x))...)) \times f_{n-1}'(f_{n-2}(...f_1(f_0(x))...)) \times... \times f_2'(f_1(f_0(x))) \times f_1'(f_0(x)) \times f_0'(x)
# $$



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
def chainAcc_OneInputTensorFunc(chain: Chain, x: Tensor)-> List[Tensor]:

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


def forwardPass_OneInputTensorFunc(chain: Chain, x:Tensor) -> List[Tensor]:
     '''Forward pass: function composition calculations while keeping the results stored.'''
     # Calculating function compositions, but not including the last function in the list.
     forwardNestings: List[Tensor] = chainAcc_OneInputTensorFunc(chain[0: len(chain) - 1], x)

     # Add x on top so result is same length as chain, for backward pass's
     # convenience.
     return  [x] + forwardNestings


def backwardPass_OneInputTensorFunc(chain: Chain, forwards: List[Tensor]) -> List[Tensor]:

     derivList: List[Tensor] = list()

     for i in list(reversed(range(0, len(chain)))):

          tensorFunc: TensorFunction = chain[i]
          forwardResult: Tensor = forwards[i]

          # Aply the chain rule
          dTensorFunc_dResult: Tensor = deriv(func = tensorFunc, inputTensor = forwardResult)

          derivList.append(dTensorFunc_dResult)

     return reduce(mul, derivList)

def chainDeriv_OneInputTensorFunc(chain: Chain, X: Tensor) -> List[Tensor]:
     # Result of the forward function composition: where n = length(chain), and:
     # f_0 = chain[0]
     # ...
     # f_n-1 = chain[n-1]
     # ... the last element in the forward compositions list tensor is f_n-1( f_n-2(... f_2 (f_1 (f_0 (x)))...))
     forwardCompositions: List[Tensor] = forwardPass_OneInputTensorFunc(chain, X)

     # Apply the chain rule: calculate derivatives and multiply them as per chain rule to get the result tensor.
     chainRuleResult: List[Tensor] = backwardPass_OneInputTensorFunc(chain, forwardCompositions)

     return chainRuleResult

# %% [markdown]
# Testing the abstract functions chainFunction and chainDeriv:
# %% codecell
x: Tensor = tensor(np.arange(-3, 8)); x
chain: List[TensorFunction] = [leakyRelu, sigmoid, square, cubic, quartic, quintic, sinT, cosT]

# %% codecell

assert torch.equal(chainDeriv_OneInputTensorFunc(chain[0:3], x), chainDerivThree(chain[0:3], x))
assert torch.equal(chainFunctions(chain[0:3], x), chainThreeFunctions(chain[0:3], x))

# %% codecell
chainDeriv_OneInputTensorFunc(chain[0:3], x)
# %% codecell
chainDerivThree(chain[0:3], x)




# %% [markdown]
# Plot the results to show the chain rule works:
# %% codecell

def plotChain(ax, chain: Chain, inputRange: Tensor) -> None:

     assert inputRange.ndim == 1, "Function requires a 1-dimensional tensor as inputRange"

     outputRange: Tensor = chainFunctions(chain = chain, x = inputRange)

     ax.plot(inputRange, outputRange)



def plotChainDeriv(ax, chain: Chain, inputRange: Tensor) -> None:
     assert inputRange.ndim == 1, "Function requires a 1-dimensional tensor as inputRange"

     outputRange: Tensor = chainDeriv_OneInputTensorFunc(chain = chain, X= inputRange)

     ax.plot(inputRange, outputRange)


# %% codecell
PLOT_RANGE: Tensor = tensor(np.arange(-3, 3, 0.01))

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






# %% [markdown]
# ## Functions with Multiple Inputs (Case: Addition)
# Defining the following function $\alpha(x,y)$ with inputs $x$ and $y$:
# $$
# a = \alpha(x, y) = x + y
# $$
# Going to feed this function through another function `sigmoid` or $\sigma$:
# $$
# s = \sigma(a) = \sigma(\alpha(x,y)) = \sigma(x + y)
# $$
# * NOTE: whenever we deal with an operation that takes multiple Tensors as input, we have to check their shapes to ensure they meet whatever conditions are required by that operation.
# * NOTE 2: below for the `multipleInputsAdd` function, all we need to check is that the shapes of `x` and `y` are identical so that addition can happen elementwise (since that is the only way it happens).
# %% codecell
def multipleInputsAdd(x: Tensor, y: Tensor, sigma: TensorFunction) -> float:
    '''(Forward pass) Function with multiple inputs and addition'''
    assert x.shape == y.shape

    a: Tensor = x + y

    return sigma(a)

# %% [markdown]
# ## Derivatives of Functions with Multiple Inputs (Case: Addition)
# Goal is to compute the derivative of each constituent function "going backward" through the computational graph and then multiply the result together to get the total derivative (as per chain rule).
# Given the function from before (calling it `f` now instead of `s`):
# $$
# f(x,y) = \sigma(\alpha(x,y)) = \sigma(x + y)
# $$
# %% [markdown]
# 1. Derivative with respect to $x$ is:
#
# Leibniz (simple) Notation:
# $$
# \frac{\partial f}{\partial x} = \frac{\partial \sigma}{\partial \alpha} \times \frac{\partial \alpha}{\partial x}
# $$
#
# Leibniz (longer) Notation:
# $$
# \frac{\partial }{\partial x}(f(x,y)) = \frac{\partial }{\partial \alpha}(\sigma(\alpha(x,y))) \times \frac{\partial }{\partial x}(\alpha(x,y))
# $$
#
# Prime Notation:
# $$
# f'(x,y) = \sigma'(\alpha(x,y)) \times \alpha'(x,y)
# $$
#
#
#
#
# 2. Derivative with respect to $y$ is:
#
# Leibniz (simple) Notation:
# $$
# \frac{\partial f}{\partial y} = \frac{\partial \sigma}{\partial \alpha} \times \frac{\partial \alpha}{\partial y}
# $$
#
# Leibniz (longer) Notation:
# $$
# \frac{\partial }{\partial y}(f(x,y)) = \frac{\partial }{\partial \alpha}(\sigma(\alpha(x,y))) \times \frac{\partial }{\partial y}(\alpha(x,y))
# $$
#
# Prime Notation:
# $$
# f'(x,y) = \sigma'(\alpha(x,y)) \times \alpha'(x,y)
# $$
# %% [markdown]
# #### Derivative of $\alpha$ function:
# $$
# \frac{\partial \alpha}{\partial x} = \frac{\partial}{\partial x}(x + y) = 1
# $$
# and same applies to the derivative with respect to $y$:
#
# $$
# \frac{\partial \alpha}{\partial y} = \frac{\partial}{\partial y}(x + y) = 1
# $$
# %% codecell
def multipleInputsAddDeriv(x: Tensor, y: Tensor,
                           sigma: TensorFunction) -> Tuple[Tensor, Tensor]:

    ''' Computes the derivative of the alpha function with respect to both inputs'''
    ### Forward pass: the simple calculation / execution of the alpha function
    a: Tensor = x + y

    ### Backward pass: computing derivatives:
    # NOTE: s = sigma = f
    ds_da: Tensor = deriv(func = sigma, inputTensor = a)
    da_dx: Tensor = 1 # same as replicating tensor 1
    da_dy: Tensor = 1
    #da_dx: Tensor = tensor([1] * len(ds_da))

    df_dx: Tensor = ds_da * da_dx
    df_dy: Tensor = ds_da * da_dy

    return df_dx, df_dy

# %% codecell
x = tensor(np.arange(-3, 8))
assert torch.equal(x, tensor([-3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7]))


y = tensor(np.arange(-5,6));  y
assert torch.equal(y, tensor([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5]))

assert x.shape == y.shape

# Sigma is a cubing function
sigma: TensorFunction = lambda tensor: tensor**3

res: Tensor = multipleInputsAdd(x, y, sigma)
assert torch.equal(res , tensor([-512, -216,  -64,   -8,  0,8,   64,  216,  512, 1000, 1728]))

# %% [markdown]
# Printing out value of the derivatives with respect to $x$ and $y$:

# %% codecell
multipleInputsAddDeriv(x, y, sigma)







# %% [markdown]
# ## Functions with Multiple Inputs (Case: Multiplication)
# Defining the following function $\beta(x,y)$ with inputs $x$ and $y$:
# $$
# \beta = \beta(x, y) = x * y
# $$
# Going to feed this function through another function `sigmoid` or $\sigma$:
# $$
# s = \sigma(\beta) = \sigma(\beta(x,y)) = \sigma(x * y)
# $$
# * NOTE: whenever we deal with an operation that takes multiple Tensors as input, we have to check their shapes to ensure they meet whatever conditions are required by that operation.
# * NOTE 2: below for the `multipleInputsMultiply` function, all we need to check is that the shapes of `x` and `y` are identical because for 1-dim tensors, multiplication happens elementwise.
# %% codecell
def multipleInputsMultiply(x: Tensor, y: Tensor, sigma: TensorFunction) -> float:
    '''(Forward pass) Function with multiple inputs and addition'''
    assert x.shape == y.shape

    beta: Tensor = x * y

    return sigma(beta)

# %% [markdown]
# ## Derivatives of Functions with Multiple Inputs (Case: Multiplication)
# Goal is to compute the derivative of each constituent function "going backward" through the computational graph and then multiply the result together to get the total derivative (as per chain rule).
# Given the function from before (calling it `f` now instead of `s`):
# $$
# f(x,y) = \sigma(\beta(x,y)) = \sigma(x * y)
# $$
#
# 1. Derivative with respect to $x$ is:
#
# Leibniz (simple) Notation:
# $$
# \frac{\partial f}{\partial x} = \frac{\partial \sigma}{\partial \beta} \times \frac{\partial \beta}{\partial x}
# $$
#
# Leibniz (longer) Notation:
# $$
# \frac{\partial }{\partial x}(f(x,y)) = \frac{\partial }{\partial \beta}(\sigma(\beta(x,y))) \times \frac{\partial }{\partial x}(\beta(x,y))
# $$
#
# Prime Notation:
# $$
# f'(x,y) = \sigma'(\beta(x,y)) \times \beta'(x,y)
# $$
#
#
#
#
# 2. Derivative with respect to $y$ is:
#
# Leibniz (simple) Notation:
# $$
# \frac{\partial f}{\partial y} = \frac{\partial \sigma}{\partial \beta} \times \frac{\partial \beta}{\partial y}
# $$
#
# Leibniz (longer) Notation:
# $$
# \frac{\partial }{\partial y}(f(x,y)) = \frac{\partial }{\partial \beta}(\sigma(\beta(x,y))) \times \frac{\partial }{\partial y}(\beta(x,y))
# $$
#
# Prime Notation:
# $$
# f'(x,y) = \sigma'(\beta(x,y)) \times \beta'(x,y)
# $$
# %% [markdown]
# #### Derivative of $\beta$ function:
# $$
# \frac{\partial \beta}{\partial x} = \frac{\partial}{\partial x}(x * y) = y
# $$
# and same applies to the derivative with respect to $y$:
#
# $$
# \frac{\partial \beta}{\partial y} = \frac{\partial}{\partial y}(x * y) = x
# $$
# %% codecell
def multipleInputsMultiplyDeriv(x: Tensor, y: Tensor,
                           sigma: TensorFunction) -> Tuple[Tensor, Tensor]:

    ''' Computes the derivative of the beta function with respect to both inputs'''
    ### Forward pass: the simple calculation / execution of the alpha function
    beta: Tensor = x * y

    ### Backward pass: computing derivatives:
    # NOTE: s = sigma = f
    ds_db: Tensor = deriv(func = sigma, inputTensor = beta)
    db_dx: Tensor = y # same as replicating tensor 1
    db_dy: Tensor = x

    df_dx: Tensor = ds_db * db_dx
    df_dy: Tensor = ds_db * db_dy

    return df_dx, df_dy

# %% codecell
x = tensor(np.arange(-3, 8))
assert torch.equal(x, tensor([-3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7]))


y = tensor(np.arange(-5,6));  y
assert torch.equal(y, tensor([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5]))

assert x.shape == y.shape

# Sigma is a cubing function
sigma: TensorFunction = lambda tensor: tensor**3

# %% codecell
beta: Tensor = x*y; beta
# %% codecell
deriv(sigma, beta)
# %% codecell
multipleInputsMultiply(x, y, sigma)





# %% [markdown]
# ## Functions with Multiple Matrix Inputs
# Here, inputs are multi-dimensional tensors, not just 1-dim tensors as in the above two examples.
#
# Define the following:
# $$
# X = \begin{pmatrix} x_1 & x_2 & ... & x_n \end{pmatrix}
# $$
#
# $$
# \begin{align}
#   W &=\begin{pmatrix}
#       w_1 \\
#       w_2 \\
#       \vdots \\
#       w_n
#   \end{pmatrix}
# \end{align}
# $$
# where $x_i, w_i \in \mathbf{R}^n$ so the elements of $X$ and $W$ themselves can also be > 0-dim tensors.
#
# Define a function to carry out the tensor multiplication between these tensors:
# $$
# \begin{align}
# N &= \nu(X, W) \\
#   &= X \times W \\
#   &= \begin{pmatrix} x_1 & x_2 & ... & x_n \end{pmatrix} \times \begin{pmatrix}  w_1 \\ w_2 \\  \vdots \\  w_n \end{pmatrix} \\
#   &= x_1 \times w_1 + x_2 \times w_2 + ... + x_n \times w_n
# \end{align}
# $$
#
# NOTE: difference between np.matmul and np.dot:
# * np.matmul: https://hyp.is/YXQQAPJoEeq_U1-RXfLGDQ/numpy.org/doc/stable/reference/generated/numpy.matmul.html
#
# * np.dot: https://hyp.is/S3p4evJoEeqQlvsyite7fg/numpy.org/doc/stable/reference/generated/numpy.dot.html
# %% codecell
def matrixForward(X: Tensor, W: Tensor) -> Tensor:
    '''Computes the forward pass of a matrix multiplication'''

    assert X.shape[1] == W.shape[0], \
        '''For matrix multiplication, with X.shape == (m, n) and W.shape == (r, p), it is required that n == r. Instead, the number of columns in the first tensor X is n == {0} and the number of rows in the second tensor is r == {1}'''.format(X.shape[1], W.shape[0])

    # matrix multiplication
    #N: Tensor = np.dot(X, W) # same as torch.matmul
    N: Tensor = torch.matmul(X, W)

    assert torch.equal(tensor(np.dot(X, W)), N)

    return N

# %% [markdown]
# ## Derivatives of Functions with Multiple Tensor Inputs (Gradient)
# Doing derivatives of tensors and matrices (2-dim tensors).
#
# Define:
# $$
# X = \begin{pmatrix} x_1 & x_2 & ... & x_n \end{pmatrix}
# $$
#
# $$
# \begin{align}
#   W &=\begin{pmatrix}
#       w_1 \\
#       w_2 \\
#       \vdots \\
#       w_n
#   \end{pmatrix}
# \end{align}
# $$
# where $x_i, w_i \in \mathbf{R}^n$ so the elements of $X$ and $W$ themselves can also be > 0-dim tensors.
#
# Define a function to carry out the tensor multiplication between these tensors:
# $$
# \begin{align}
# N &= \nu(X, W) \\
#   &= X \times W \\
#   &= \begin{pmatrix} x_1 & x_2 & ... & x_n \end{pmatrix} \times \begin{pmatrix}  w_1 \\ w_2 \\  \vdots \\  w_n \end{pmatrix} \\
#   &= x_1 \times w_1 + x_2 \times w_2 + ... + x_n \times w_n
# \end{align}
# $$

# %% [markdown]
# #### Calculating Derivative with Respect to a Matrix (Gradient):
# Since each $\frac {\partial \nu} {\partial x_i} = \frac{\partial}{\partial x_i}(\nu(X, W)) = \frac{\partial}{\partial x_i} (x_1 \times w_1 + x_2 \times w_2 + ... + x_i \times w_i + ... + x_n \times w_n) = w_i$,
#
# we have that $\frac{\partial \nu}{\partial X}$ is:
#
# $$
# \begin{align}
# \frac{\partial \nu}{\partial X}
#   &= {\large \begin{pmatrix} \frac{\partial \nu}{\partial x_1} & \frac{\partial \nu}{\partial x_2} & ... & \frac{\partial \nu}{\partial x_n}  \end{pmatrix} } \\
#   &= \begin{pmatrix} w_1 & w_2 & ... & w_n \end{pmatrix} \\
#   &= W^T
# \end{align}
# $$
# Similarly:
# $$
# \frac{\partial \nu}{\partial W}
#   = {\large \begin{pmatrix} \frac{\partial \nu}{\partial w_1} \\ \frac{\partial \nu}{\partial w_2} \\ \vdots \\ \frac{\partial \nu}{\partial w_n}  \end{pmatrix} }
#   = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}
#   = X^T
# $$
# Therefore,
# $$
# \frac{\partial \nu}{\partial X} = W^T
# $$
#
# $$
# \frac{\partial \nu}{\partial W} = X^T
# $$
# %% codecell
def matrixBackward_X(X: Tensor, W: Tensor) -> Tensor:
    '''Computes the backward pass of a matrix multiplication with respect to the first argument.'''

    ### Backward pass
    dN_dX: Tensor = torch.transpose(W, 1, 0)

    return dN_dX
# %% [markdown]
# Checking the numpy transpose correspondence with torch transpose:
# %% codecell
X: Tensor = torch.arange(2*3*4).reshape(3,2,4)
W: Tensor = torch.arange(5*3*2).reshape(5,3,2)
dN_dX: Tensor = matrixBackward_X(X, W)
dN_dX.shape

# %% codecell
# Assert: No matter which order we specify dimensions in (1,0) or (0,1) since they jsut refer to the dimensions, no order necessary.
assert torch.equal(dN_dX, torch.transpose(W, 0, 1))
# Assert: numpy transpose equals torch transpose
### NOTE: in numpy transpose, we need to specify ALL dimension IDs (1,0,2) even when just transposing through
# some of the  dimensions (1,0).
assert torch.equal(tensor(dN_dX.numpy().transpose((1,0,2))), torch.transpose(dN_dX,1,0))


# %% [markdown]
# ## Vector Functions with Multiple Tensor Inputs (Extra Output Function)
# Using $N = \nu(X, W)$ from previously, define the following function $s = f(X, W)$ which passes $\nu$ through an extra function $\sigma$::
# $$
# s = f(X, W) = \sigma(\nu(X, W)) = \sigma(x_1 \times w_1 + ... + x_n \times w_n)
# $$

# %% codecell

def matrixForwardSigma(X: Tensor, W: Tensor, sigma: TensorFunction) -> Tensor:
    ''' Computes the forward pass of a function involving matrix multiplication and one extra output function'''
    # assert X.shape[1] == W.shape[0] # NOTE: this is only for vector inputs, if we want higher-dim inputs then the second-to last dim of W should equal the first dim of X.
    isFirstPartEqualShape: bool = X.shape[:len(X.shape)-2] == W.shape[0:len(W.shape)-2]
    canDoMatMul: bool = X.shape[-1] == W.shape[-2]

    assert isFirstPartEqualShape and canDoMatMul

    # Matrix multiplication:
    N: Tensor = torch.matmul(X, W)

    # Feeding the output of the matrix multiplication through sigma:
    S: Tensor = sigma(N)

    return S

# %% codecell
# Testing requirements for matrx multiplication: need all the dimensions the same, except the second last one of X and last one of W:
X = torch.arange(7*8*2*4*5).reshape(2,8,5,7,4)
W = torch.arange(3*8*2*4*5).reshape(2,8,5,4,3)
sigma: TensorFunction = lambda t: t**3 + t


assert matrixForwardSigma(X, W, sigma).shape == (2, 8, 5, 7, 3)

X: Tensor = torch.arange(3*4*2*2).reshape(3,2,4,2)
W: Tensor = torch.arange(3*4*2*2).reshape(3,2,2,4)
sigma: TensorFunction = lambda t: t**3 + t


assert matrixForwardSigma(X, W, sigma).shape == (3, 2, 4, 4)



# %% [markdown]
# ## Derivative of Functions with Multiple Tensor Inputs (Extra Function)
# #### Abstract Calculation:
# Since $s = f(X, W) = \sigma(\nu(X, Y))$, we apply the chain rule to find $\frac{\partial f}{\partial X}$:
#
# $$
# \frac{\partial f}{\partial X}
#   = \frac{\partial \sigma}{\partial \nu} \times \frac{\partial \nu}{\partial X}
#   = \frac{\partial \sigma}{\partial \nu} \times W^T
# $$
#
# #### Evaluation Calculation:
# $$
# \begin{align}
# \frac{\partial f}{\partial X} \bigg|_{\large \nu = x_1 \times w_1 + ... + x_n \times w_n}
#   &= \frac{\partial \sigma }{\partial \nu} \bigg|_{\large \nu = x_1 \times w_1 + ... + x_n \times w_n} \times \frac{\partial \nu}{\partial X}  \\
#   &= \frac{\partial }{\partial \nu}\sigma(\nu) \bigg|_{\large \nu = x_1 \times w_1 + ... + x_n \times w_n} \times W^T \\
#   &= \frac{\partial}{\partial (x_1 \times w_1 + ... + x_n \times w_n)}(\sigma(x_1 \times w_1 + ... + x_n \times w_n)) \times W^T
# \end{align}
# $$

# %% codecell
def matrixBackwardSigma_X(X: Tensor, W: Tensor, sigma: TensorFunction) -> Tensor:
    '''Computes derivative of matrix function with respect to the first element X'''

    isFirstPartEqualShape: bool = X.shape[:len(X.shape)-2] == W.shape[0:len(W.shape)-2]
    canDoMatMul: bool = X.shape[-1] == W.shape[-2]

    assert isFirstPartEqualShape and canDoMatMul

    ### Forward pass: Matrix multiplication:
    # X.shape = (...,n, m)
    # W.shape = (...,m, p)
    N: Tensor = torch.matmul(X, W)
    # N.shape = (...,n, p)

    # Feeding the output of the matrix multiplication through sigma:
    S: Tensor = sigma(N)
    # S.shape = N.shape = (...,n, p)



    ### Backward pass: chain rule for matrix (df/dX) or (dS/dX)
    dS_dN: Tensor = deriv(sigma, N)
    # dS_dN.shape = N.shape = (...,n, p)

    # Transpose along first two dimensions (this example assumes we are
    # using 2-dim tensors which are matrices)
    lastDimIndex: int = W.ndim - 1
    secLastDimIndex: int = W.ndim - 2

    # NOTE: need to convert to float tensor because dS_dN is a float tensor, after the approx derivative calculation,
    # while W is just a long tensor and if we don't convert, we get runtime error. .
    dN_dX: Tensor = torch.transpose(W, secLastDimIndex, lastDimIndex).type(torch.FloatTensor)
    # torch.transpose(W, 1, 0)
    ## dN_dX.shape = W^T.shape = (...,p,m)


    # TODO: is the chain rule here always matrix multiplication or is it dot product (np.dot) as in the book (page 37)?
    dS_dX: Tensor = torch.matmul(dS_dN, dN_dX)

    assert dS_dX.shape == X.shape

    return dS_dX ## shape == (...,n,m)


# %% codecell

X: Tensor = torch.arange(3*4*2*7).reshape(3,2,4,7)
W: Tensor = torch.arange(3*5*2*7).reshape(3,2,7,5)
sigma: TensorFunction = lambda t: t**3 + t


assert matrixBackwardSigma_X(X, W, sigma).shape == (3, 2, 4, 7)

# %% codecell
x: Tensor = torch.rand(2,10)
w: Tensor = torch.rand(10,2)

matrixBackwardSigma_X(x, w, sigmoid)
# %% [markdown]
# #### Testing if the derivatives computed are correct:
# A simple test is to perturb the array and observe the resulting change in output. If we increase $x_{2,1,3}$ by 0.01 from -1.726 to -1.716 we should see an increase in the value porduced by the forward function of the *gradient of the output with respect to $x_{2,1,3}$*.
# %% codecell

def doForwardSigmaIncr(X: Tensor, W: Tensor, sigma: TensorFunction, indices: Tuple[int], increment: float) -> Tensor:

    X[indices] = -1.726 # setting the starting value for sake of example
    X_ = X.clone()

    # Increasing the value at that point by 0.01
    X_[indices] = X[indices] + increment

    assert X[indices] == -1.726
    assert X_[indices] == X[indices] + increment

    return matrixForwardSigma(X_, W, sigma)


# %% [markdown]
# Testing with 2-dim tensors:
# %% codecell
X: Tensor = torch.arange(5*4).reshape(5,4).type(torch.FloatTensor)
W: Tensor = torch.rand(4,5)

indices = (2,1)
increment = 0.01
inc: Tensor = doForwardSigmaIncr(X, W, sigma, indices = indices, increment = increment)
incNot: Tensor = doForwardSigmaIncr(X, W, sigma, indices = indices, increment = 0)

print(torch.sum((inc - incNot) / increment))

print(matrixBackwardSigma_X(X, W, sigma)[indices])
# %% [markdown]
# Testing with 3-dim tensors:

# %% codecell
X: Tensor = torch.arange(5*4*3).reshape(5,4,3).type(torch.FloatTensor)
W: Tensor = torch.rand(5,3,4)

indices = (2,1,2)
increment = 0.01
inc: Tensor = doForwardSigmaIncr(X, W, sigma, indices = indices, increment = increment)
incNot: Tensor = doForwardSigmaIncr(X, W, sigma, indices = indices, increment = 0)

print(torch.sum((inc - incNot) / increment))

print(matrixBackwardSigma_X(X, W, sigma)[indices])


# %% [markdown]
# Testing with 4-dim tensors:

# %% codecell
X: Tensor = torch.arange(5*4*3*2).reshape(5,2,4,3).type(torch.FloatTensor)
W: Tensor = torch.rand(5,2,3,1)

indices = (2,1,2,0)
increment = 0.01
inc: Tensor = doForwardSigmaIncr(X, W, sigma, indices = indices, increment = increment)
incNot: Tensor = doForwardSigmaIncr(X, W, sigma, indices = indices, increment = 0)

print(torch.sum((inc - incNot) / increment))

print(matrixBackwardSigma_X(X, W, sigma)[indices])















# %% [markdown]
# ## Functions with Multiple Matrix Inputs: (Forward Pass) Lambda Sum
# Defining variable-element matrices $X \in \mathbb{R}^{n \times m}$ and $W \in \mathbb{R}^{m \times p}$ so that $X$ and $W$ are now 2D matrices instead of higher-dim tensors (letting $X$ be $n \times m$ and $W$ be $m \times p$):
# $$
# X = \begin{pmatrix}
# x_{11} & x_{12} & ... & x_{1m} \\
# x_{21} & x_{22} & ... & x_{2m} \\
# \vdots & \vdots & ... & \vdots \\
# x_{n1} & x_{n2} & ... & x_{nm}
# \end{pmatrix}
# $$
#
# $$
# W = \begin{pmatrix}
# w_{11} & w_{12} & ... & w_{1p} \\
# w_{21} & w_{22} & ... & w_{2p} \\
# \vdots & \vdots & ... & \vdots \\
# w_{m1} & w_{m2} & ... & w_{mp}
# \end{pmatrix}
# $$



# %% [markdown]
# Define some straightforward operations on these matrices:
#
# ### STEP 1: Defining $N = \nu(X, W) = X \times W$
# 
# Let
# * $\nu : \mathbb{R}^{(n \times m) \times (m \times p)} \rightarrow \mathbb{R}^{n \times p}$
# * $N \in \mathbb{R}^{n \times p}$
# 
# 
# 
# Multiply the matrices $X$ and $W$ together as the function $N = \nu(X, W)$. Denote the row $i$ and column $j$ in the resulting matrix as $(XW)_{ij}$:
#
# $$
# \begin{aligned}
# N &= \nu(X,W) \\
# &= X \times W \\
# &= \begin{pmatrix}
#   x_{11} \cdot w_{11} + x_{12} \cdot w_{21} + ... + x_{1m} \cdot w_{m1} &
#   x_{11} \cdot w_{12} + x_{12} \cdot w_{22} + ... + x_{1m} \cdot w_{m2} &
#   ... &
#   x_{11} \cdot w_{1p} + x_{12} \cdot w_{2p} + ... + x_{1m} \cdot w_{mp} \\
#   x_{21} \cdot w_{11} + x_{22} \cdot w_{21} + ... + x_{2m} \cdot w_{m1} &
#   x_{21} \cdot w_{12} + x_{22} \cdot w_{22} + ... + x_{2m} \cdot w_{m2} &
#   ... &
#   x_{21} \cdot w_{1p} + x_{22} \cdot w_{2p} + ... + x_{2m} \cdot w_{mp} \\
#   \vdots & \vdots & \vdots & \vdots \\
#   x_{n1} \cdot w_{11} + x_{n2} \cdot w_{21} + ... + x_{nm} \cdot w_{m1} &
#   x_{n1} \cdot w_{12} + x_{n2} \cdot w_{22} + ... + x_{nm} \cdot w_{m2} &
#   ... &
#   x_{n1} \cdot w_{1p} + x_{n2} \cdot w_{2p} + ... + x_{nm} \cdot w_{mp} \\
# \end{pmatrix} \\
# &= \begin{pmatrix}
#   (XW)_{11} &   (XW)_{12} & ... & (XW)_{1p} \\
#   (XW)_{21} &   (XW)_{22} & ... & (XW)_{2p} \\
#   \vdots & \vdots & \vdots & \vdots \\
#   (XW)_{n1} &   (XW)_{n2} & ... & (XW)_{np}
# \end{pmatrix}
# \end{aligned}
# $$
# where `X.shape == (n, m)`, and `W.shape == (m, p)`, and `N.shape == (n, p)`.
#




# %% [markdown]
# ### STEP 2: Defining $S = \sigma_{\text{apply}}(N)$
#
# Assume that $\sigma_{\text{apply}} : \mathbb{R}^{n \times p} \rightarrow \mathbb{R}^{n \times p}$ while $\sigma : \mathbb{R} \rightarrow \mathbb{R}$, so the function $\sigma_{\text{apply}}$ takes in a matrix and returns a matrix while the simple $\sigma$ acts on the individual elements $N_{ij} = XW_{ij}$ in the matrix argument $N$ of $\sigma_{\text{apply}}$.
#
# * $\sigma : \mathbb{R} \rightarrow \mathbb{R}$
# * $\sigma_\text{apply} : \mathbb{R}^{n \times p} \rightarrow \mathbb{R}^{n \times p}$
# * $S \in \mathbb{R}^{n \times p}$
#
# 
# Then we can define $S$ by feeding $N$ through some differentiable function $\sigma$ (just applying $\sigma$ to every element of the matrix operation defined by $N$):
# $$
# \begin{aligned}
# S &= \sigma(N) \\
#  &= \sigma(\nu(X, W)) \\
#  &= \sigma(X \times W) \\
#  &= \begin{pmatrix}
#   \sigma \Big( x_{11} \cdot w_{11} + x_{12} \cdot w_{21} + ... + x_{1m} \cdot w_{m1} \Big) &
#   \sigma \Big( x_{11} \cdot w_{12} + x_{12} \cdot w_{22} + ... + x_{1m} \cdot w_{m2} \Big) &
#   ... &
#   \sigma \Big( x_{11} \cdot w_{1p} + x_{12} \cdot w_{2p} + ... + x_{1m} \cdot w_{mp} \Big) \\
#   \sigma \Big( x_{21} \cdot w_{11} + x_{22} \cdot w_{21} + ... + x_{2m} \cdot w_{m1} \Big) &
#   \sigma \Big( x_{21} \cdot w_{12} + x_{22} \cdot w_{22} + ... + x_{2m} \cdot w_{m2} \Big) &
#   ... &
#   \sigma \Big( x_{21} \cdot w_{1p} + x_{22} \cdot w_{2p} + ... + x_{2m} \cdot w_{mp} \Big) \\
#   \vdots & \vdots & \vdots & \vdots \\
#   \sigma \Big( x_{n1} \cdot w_{11} + x_{n2} \cdot w_{21} + ... + x_{nm} \cdot w_{m1} \Big) &
#   \sigma \Big( x_{n1} \cdot w_{12} + x_{n2} \cdot w_{22} + ... + x_{nm} \cdot w_{m2} \Big) &
#   ... &
#   \sigma \Big( x_{n1} \cdot w_{1p} + x_{n2} \cdot w_{2p} + ... + x_{nm} \cdot w_{mp} \Big)
# \end{pmatrix} \\
# &= \begin{pmatrix}
#   \sigma \Big( (XW)_{11} \Big) &   \sigma \Big( (XW)_{12} \Big) & ... & \sigma \Big( (XW)_{1p} \Big) \\
#   \sigma \Big( (XW)_{21} \Big) &   \sigma \Big( (XW)_{22} \Big) & ... & \sigma \Big( (XW)_{2p} \Big) \\
#   \vdots & \vdots & \vdots & \vdots \\
#   \sigma \Big( (XW)_{n1} \Big) &   \sigma \Big( (XW)_{n2} \Big) & ... & \sigma \Big( (XW)_{np} \Big)
# \end{pmatrix}
# \end{aligned}
# $$
#
# where `S.shape == N.shape == (n, p)`.







# %% [markdown]
# ### STEP 3: Defining $L = \Lambda(S)$
# 
# Defining a $\Lambda$ function to sum up the elements in the matrix to find the total effect of changing each element of a matrix. 
# 
# Assume:
# * $\Lambda: \mathbb{R}^{n \times p} \rightarrow \mathbb{R}$
# * $L \in \mathbb{R}$
#
# 
# $$
# \begin{aligned}
# L &= \Lambda(S) \\
#   &= \Lambda(\sigma_\text{apply}(N(X, W))) \\
#   &= \Lambda \begin{pmatrix}
# \begin{pmatrix}
#   \sigma \Big( (XW)_{11} \Big) &   \sigma \Big( (XW)_{12} \Big) & ... & \sigma \Big( (XW)_{1p} \Big) \\
#   \sigma \Big( (XW)_{21} \Big) &   \sigma \Big( (XW)_{22} \Big) & ... & \sigma \Big( (XW)_{2p} \Big) \\
#   \vdots & \vdots & \vdots & \vdots \\
#   \sigma \Big( (XW)_{n1} \Big) &   \sigma \Big( (XW)_{n2} \Big) & ... & \sigma \Big( (XW)_{np} \Big)
# \end{pmatrix}
# \end{pmatrix} \\
#
# &= \sigma(XW_{11}) + ... + \sigma(XW_{1p}) + \sigma(XW_{21}) + ... + \sigma(XW_{2p}) + ... ... ... + \sigma(XW_{n1}) + ... + \sigma(XW_{np})
# \end{aligned}
# $$
#  where $\Lambda$.`shape == (1,1)` so $\Lambda$ is a constant.
#
# 
# * NOTE: the code can handle higher-dim tensors while the formulas above handle only 2-dim tensors (matrices). So in the code, we have:
#   * `X.shape == (..., n, m)`
#   * `W.shape == (..., m, p)`
#   * `N.shape == (..., n, p)`
#   * `S.shape == (..., n, p)`
#   * $\Lambda$`.shape == (1,1)` 
# 
# and in the formulas we have:
#   * `X.shape == (n, m)`
#   * `W.shape == (m, p)`
#   * `N.shape == (n, p)`
#   * `S.shape == (n, p)`
#   * $\Lambda$`.shape == (1,1)`
#
# 
# * NOTE: the '...' is set to mean that all dimensions before the last two are the same for all tensors in these calculations.



# %% codecell
def matrixForwardSum(Xa: Tensor, Wa: Tensor, sigma: TensorFunction) -> float:
     '''Computes the result of the forward pass of the function L with input tensors X and W and function sigma

     X.shape == (..., n, m)
     W.shape == (..., m, p)
     '''

     # NOTE: cast to float in case X or W is float else error   gets thrown
     X: Tensor = Xa.clone()
     W: Tensor = Wa.clone()
     # Now X and W are either float or long tensor

     if(Xa.type() == 'torch.FloatTensor' and Wa.type() == 'torch.LongTensor'):
         W: FloatTensor = W.type(torch.FloatTensor)
     elif(Xa.type() == 'torch.LongTensor' and Wa.type() == 'torch.FloatTensor'):
         X: FloatTensor = X.type(torch.FloatTensor)

     # Now X, W are either BOTH Long or BOTH Float tensors.

     isFirstPartEqualShape: bool = X.shape[:len(X.shape)-2] == W.shape[0:len(W.shape)-2]
     canDoMatMul: bool = X.shape[-1] == W.shape[-2]

     assert isFirstPartEqualShape and canDoMatMul

     # Matrix multiplication:
     N: Tensor = torch.matmul(X, W)
     ## N.shape == (..., n,p)

     # Feeding the output of the matrix multiplication through sigma:
     S: Tensor = sigma(N)
     ## S.shape == (..., n, p)

     # Sum all the elements : L = lambda(S(N(X,Y)))
     L: Tensor = torch.sum(S)
     ## L.shape == (1,1) (0 - dim tensor or constant)

     return L # shape 1x1









# %% [markdown]
# ## Derivative of Functions with Multiple Matrix Inputs: (Backward Pass) Lambda Sum
#
# We have a number $L$ and we want to find out the gradient of $L$ with respect to $X$ and $W$; how much changing *each element* of these input matrices (so each $x_{ij}$ and each $w_{ij}$) would change $L$. This is written as:
#
# ### Direct Way:
#
# **Derivative with respect to $X$: **
# 
# $$
# \large
# \frac{\partial \Lambda}{\partial X} = \begin{pmatrix}
#   \frac{\partial \Lambda}{\partial x_{11}} & \frac{\partial \Lambda}{\partial x_{12}} & ... & \frac{\partial \Lambda}{\partial x_{1m}} \\
#   \frac{\partial \Lambda}{\partial x_{21}} & \frac{\partial \Lambda}{\partial x_{22}} & ... & \frac{\partial \Lambda}{\partial x_{2m}} \\
#   \vdots & \vdots & \vdots & \vdots \\
#   \frac{\partial \Lambda}{\partial x_{n1}} & \frac{\partial \Lambda}{\partial x_{n2}} & ... & \frac{\partial \Lambda}{\partial x_{nm}}
# \end{pmatrix}
# $$
#
# 
# **Derivative with respect to $W$: **
# 
# $$
# \large
# \frac{\partial \Lambda}{\partial W} = \begin{pmatrix}
#   \frac{\partial \Lambda}{\partial w_{11}} & \frac{\partial \Lambda}{\partial w_{12}} & ... & \frac{\partial \Lambda}{\partial w_{1p}} \\
#   \frac{\partial \Lambda}{\partial w_{21}} & \frac{\partial \Lambda}{\partial w_{22}} & ... & \frac{\partial \Lambda}{\partial w_{2p}} \\
#   \vdots & \vdots & \vdots & \vdots \\
#   \frac{\partial \Lambda}{\partial w_{m1}} & \frac{\partial \Lambda}{\partial w_{m2}} & ... & \frac{\partial \Lambda}{\partial w_{mp}}
# \end{pmatrix}
# $$
#
# 
# 
# 
# ### Chain Rule way (Backward pass):
# 
# The chain rule gives the same result as the direct way above but is simpler for the user to calculate:
# 
# #### Preliminary Variable Set-up: 
# %% codecell
n,m,p = 3,3,2

# Variables
X = Matrix(n, m, lambda i,j : var_ij('x', i, j))
W = Matrix(m, p, lambda i,j : var_ij('w', i, j))
A = MatrixSymbol('X',n,m)
B = MatrixSymbol('W',m,p)
# matrix variable for sympy Lambda function arguments
M = MatrixSymbol('M', i, j)# abstract shape


# N function
v = Function("nu",applyfunc=True)
v_ = lambda a,b: a*b
vL = Lambda((a,b), a*b)
VL = Lambda((A,B), MatrixSymbol('V', A.shape[0], B.shape[1]))
vN = lambda mat1, mat2: Matrix(mat1.shape[0], mat2.shape[1], lambda i, j: Symbol("n_{}{}".format(i+1, j+1))); vN

Nelem = vN(X, W)
Nspec = v_(X,W)
N = v(A,B)


# S function
sigma = Function('sigma')
sigmaApply = Function("sigma_apply") #lambda matrix:  matrix.applyfunc(sigma)
sigmaApply_ = lambda matrix: matrix.applyfunc(sigma)
sigmaApply_L = Lambda(M, M.applyfunc(sigma))

S = sigmaApply(N)
Sspec = S.subs({A:X, B:W}).replace(v, v_).replace(sigmaApply, sigmaApply_)
Selem = S.replace(v, vN).replace(sigmaApply, sigmaApply_)



# L function
lambd = Function("lambda")
lambd_ = lambda matrix : sum(matrix)
# Declaring appropriate matrix shape so that it is argument of lambda function
ABres = MatrixSymbol("AB", A.shape[0], B.shape[1])
lambd_L = Lambda(ABres, sum(ABres))

L = compose(lambd, sigmaApply, v)(A, B)
L


# %% [markdown]
# ### Chain Rule Derivative with respect to $X$
# 
# **Derivative with respect to $X$: The Abstract Way**
#
# $$
# \begin{aligned}
# \frac{\partial L}{\partial X} &= \bigg( \frac{\partial L}{\partial S} \odot  \frac{\partial S}{\partial N} \bigg) \times \frac{\partial N}{\partial X}  \\
# &= \bigg( \frac{\partial L}{\partial S} \odot \frac{\partial S}{\partial N} \bigg) \times W^T 
# \end{aligned}
# $$
# where $\odot$ signifies the Hadamard product and $\times$ is matrix multiplication.

# %%
dL_dX_overallAbstract = compose(lambd, sigmaApply)(VL).diff(A).replace(VL, v(A, B))

dL_dX_overallAbstract




# %% [markdown]
# **Derivative with respect to $X$: The Matrix-Derivative Abstract Way**
# %%
dL_dX_abstract = compose(lambd, sigmaApply, v)(A, B).replace(v, v_).replace(sigmaApply, sigmaApply_).diff(A)
dL_dX_abstract



# %% [markdown]
# **Derivative with respect to $X$: The Step by Step Way**
# %%
# Step by step version
unapplied = sigmaApply_L(vN(A,B))
applied = unapplied.doit()

showGroup([
     unapplied, applied
])

# %%
dL_dX_step = compose(lambd, sigmaApply, v)(A,B).replace(v, v_).replace(sigmaApply, sigmaApply_).diff(A).subs({A*B : vN(A,B)}).doit()

showGroup([
    dL_dX_step,
    dL_dX_step.replace(unapplied, applied),
    dL_dX_step.subs({B:W}).doit(),
    dL_dX_step.subs({B:W}).doit().replace(unapplied, applied)
])


# %% [markdown]
# **Derivative with respect to $X$: The Hadamard Way**

# %% 
# BUIDLING up the pieces manually: 
dL_dS = lambd(Selem).replace(lambd, lambd_L).diff(Selem)

dS_dN = compose(sigmaApply)(M).replace(sigmaApply, sigmaApply_).diff(M).subs({M : vN(A,B)}).doit()


# Observe how X^T and W^T end up in different spots. 
# TODO show this with the matrix diff calculuator library


dN_dX = B.transpose()

dS_dX = dS_dN * dN_dX
dS_dX_abstract = compose(sigmaApply, v)(A,B).replace(v, v_).replace(sigmaApply, sigmaApply_).diff(A)

dL_dN = HadamardProduct(dL_dS, dS_dN)

# This is the end product, the above pieces need to be combined properly to yield this result below, testing with assert...
dL_dX = dL_dN * dN_dX #).subs(B, W).doit()
dL_dX_hadamard = dL_dX.subs(B, W).doit()

assert dL_dX == HadamardProduct(dL_dS, dS_dN) * dN_dX
# %%
showGroup([
    dL_dS, 
    dL_dN, 
    dS_dN
])
# %%
showGroup([
    dS_dX, 
    dS_dX.subs(B, W),
    dS_dX_abstract
])
# %%
showGroup([
    dL_dX, 
    dL_dX_hadamard
])




# %% [markdown]
# ### Chain Rule Derivative with respect to $W$:
#
# **Derivative with respect to $W$: The Abstract Way**
#
# $$
# \begin{aligned}
# \frac{\partial L}{\partial W} &= \frac{\partial N}{\partial W} \times \bigg( \frac{\partial L}{\partial S} \odot \frac{\partial S}{\partial N} \bigg) \\
# &= X^T \times \bigg( \frac{\partial L}{\partial S} \odot \frac{\partial S}{\partial N} \bigg)
# \end{aligned}
# $$
# where $\odot$ signifies the Hadamard product and $\times$ is matrix multiplication.
# %%
dL_dW_overallAbstract = compose(lambd, sigmaApply)(VL).diff(B).replace(VL, v(A, B))

dL_dW_overallAbstract


# %% [markdown]
# **Derivative with respect to $W$: The Matrix-Derivative Abstract Way**
# %%
dL_dW_abstract = compose(lambd, sigmaApply, v)(A, B).replace(v, v_).replace(sigmaApply, sigmaApply_).diff(B)
dL_dW_abstract



# %% [markdown]
# **Derivative with respect to $W$: The Step by Step Way**
# %%
unapplied = sigmaApply_L(vN(A,B))
applied = unapplied.doit()

showGroup([
     unapplied,
     applied
])

# %%
dL_dW_step = compose(lambd, sigmaApply, v)(A,B).replace(v, v_).replace(sigmaApply, sigmaApply_).diff(B).subs({A*B : vN(A,B)}).doit()

showGroup([
    dL_dW_step,
    dL_dW_step.replace(unapplied, applied),
    # Carrying out the multplication:
    dL_dW_step.subs({A:X}).doit(), # replace won't work here
    dL_dW_step.subs({A:X}).doit().replace(unapplied, applied)
])



# %% [markdown]
# **Derivative with respect to $W$: The Hadamard Way**

# %%
dN_dW = A.transpose()

dS_dW = dN_dW * dS_dN
dS_dW_abstract = compose(sigmaApply, v)(A,B).replace(v, v_).replace(sigmaApply, sigmaApply_).diff(B)

# This is the end product, the above pieces need to be combined properly to yield this result below, testing with assert...
dL_dW = HadamardProduct(dL_dS, dS_dW)
dL_dW_hadamard = dL_dW.subs(A,X).doit()

assert dL_dW == HadamardProduct(dL_dS, dN_dW * dS_dN )

# %%
showGroup([
    dS_dW, 
    dS_dW.subs(A, X).doit(), 
    dS_dW_abstract
])
# %%
showGroup([
     dL_dS,
    dL_dW, 
    dL_dW_hadamard
])



# %% [markdown]
# ### Chain Rule Derivative in Code for $\frac{\partial L}{\partial X}$: 
#
# $$
# \begin{aligned}
# \frac{\partial L}{\partial X} &= \bigg( \frac{\partial L}{\partial S} \odot  \frac{\partial S}{\partial N} \bigg) \times \frac{\partial N}{\partial X}  \\
# &= \bigg( \frac{\partial L}{\partial S} \odot \frac{\partial S}{\partial N} \bigg) \times W^T 
# \end{aligned}
# $$
# where $\odot$ signifies the Hadamard product and $\times$ is matrix multiplication.
# %% codecell

# TODO: equivalent for the matrix W (understand why dot first then matmul in the chain rule)


def matrixBackwardSum_X(Xa: Tensor, Wa: Tensor, sigma: TensorFunction) -> Tensor:
    '''Computes derivative of matrix function with respect to the first element X'''

    # NOTE: cast to float in case X or W is float else error   gets thrown
    X: Tensor = Xa.clone()
    W: Tensor = Wa.clone()
    # Now X and W are either float or long tensor

    TYPE_CAST = ['torch.FloatTensor', 'torch.DoubleTensor']

    if(Xa.type() in TYPE_CAST and Wa.type() == 'torch.LongTensor'):
         W = W.type(Xa.type())
    elif(Xa.type() == 'torch.LongTensor' and Wa.type() in TYPE_CAST):
         X = X.type(Wa.type())

    # Now X, W are either BOTH Long or BOTH Float tensors.


    isFirstPartEqualShape: bool = X.shape[:len(X.shape)-2] == W.shape[0:len(W.shape)-2]
    canDoMatMul: bool = X.shape[-1] == W.shape[-2]

    assert isFirstPartEqualShape and canDoMatMul

    ### Forward pass: Matrix multiplication:
    # X.shape = (...,n, m)
    # W.shape = (...,m, p)
    N: Tensor = torch.matmul(X, W)
    # N.shape = (...,n, p)

    # NOTE: result of matmul is either Float or Long Tensor depending on whether both X,W are Float or Long tensor. But they must be of the same kind (hence the if-else above)

    # Feeding the output of the matrix multiplication through sigma:
    S: Tensor = sigma(N)
    # S.shape = N.shape = (...,n, p)

    # Sum all the elements:
    #L: Tensor = torch.sum(S)
    # L.shape == (1 x 1)



    ### Backward pass: chain rule for matrix (df/dX) or (dS/dX)
    dL_dS: FloatTensor = torch.ones(S.shape)
    # dL_dS.shape == S.shape == (...,n, p)

    dS_dN = deriv(sigma, N) # NOTE: result is always a float tensor since deriv() is an approximation.
    # dS_dN.shape = N.shape = (...,n, p)
    dS_dN: FloatTensor = dS_dN.type(torch.FloatTensor) #  in case it is DoubleTensor so that no error when multiplying dlds and dsdn

    # TODO FIGURE OUT why we have element-wise multiplication here (Hadamard product):
    dL_dN: FloatTensor = dL_dS * dS_dN
    # dL_dN.shape == (...,n, p)
    ## NOTE: Matrix multiplication with "*" is allowed even when the tensors have different types, so no need to convert dL_dS to type Float here, to match type float of dS_dN

    # Transpose along first two dimensions (this example assumes we are
    # using 2-dim tensors which are matrices)

    # NOTE: need to convert to float tensor because dS_dN is a float tensor, after the approx derivative calculation,
    # while W is just a long tensor and if we don't convert, we get runtime error. .
    dN_dX: FloatTensor = torch.transpose(W, W.ndim - 2, W.ndim - 1).type(torch.FloatTensor)
    # torch.transpose(W, 1, 0)
    ## dN_dX.shape = W^T.shape = (...,p,m)


    # TODO: why matrix multiplication here and why element wise multiply above?
    # Hadamard elementwise product (*) to get dL_dN then matrix multiply (torch.matmul) by dN_dX
    dL_dX: FloatTensor = torch.matmul( (dL_dS * dS_dN) , dN_dX)
    # dL_dX.shape == (..., n,m)

    assert dL_dX.shape == X.shape

    return dL_dX ## shape == (...,n,m)



# %% [markdown]
# #### Testing if the derivatives computed are correct:
# A simple test is to perturb the array and observe the resulting change in output. If we increase $x_{i,j,k}$ by 0.001  we should see an increase in the value porduced by the forward function of the *gradient of the output with respect to $x_{i,j,k}$*.
# %% codecell

def doForwardSumIncr_X(Xa: Tensor, Wa: Tensor, sigma: TensorFunction, indices: Tuple[int]) -> Tensor:

     ## WARNING: the X must be FloatType tensors or else the later assertions here will fail! (only integer part of decimal gets copied)

     # NOTE: cast to float in case X or W is float else error   gets thrown
     X: Tensor = Xa.clone()
     W: Tensor = Wa.clone()
     # Now X and W are either float or long tensor

     TYPE_CAST = ['torch.FloatTensor', 'torch.DoubleTensor']

     if(Xa.type() in TYPE_CAST and Wa.type() == 'torch.LongTensor'):
          W = W.type(Xa.type())
     elif(Xa.type() == 'torch.LongTensor' and Wa.type() in TYPE_CAST):
          X = X.type(Wa.type())

     # Now X, W are either BOTH Long or BOTH Float tensors.
     #print("BEFORE: {}".format(X[indices]))
     
     Xclone = X.clone()

     # Creating the increment to be same size as the X[indices] shape: 
     increment = torch.tensor([0.001]).repeat_interleave(Xclone[indices].numel()).reshape(Xclone[indices].shape)

     # Increasing the value at that point by 0.01
     Xclone[indices] = X[indices] + increment

     #print("AFTER: {}".format(Xclone[indices]))
     #assert X[indices] == FLAG_NUM
     #assert Xclone[indices] == X[indices] + increment

     return ( (matrixForwardSum(Xclone, W, sigma) - matrixForwardSum(X, W, sigma)) / increment ).type(torch.FloatTensor)




# %% [markdown]
# **Testing with 2-dim tensors:**
# 
# Can see that test is successful. The `dLdX[indices]` is approximately equal the `forwardIncr`:
# %% codecell
np.random.seed(190204)


X: Tensor = Tensor(np.random.randn(3,3))
W: Tensor = Tensor(np.random.randn(3,2))

#sigma: TensorFunction = lambda t: 2*t + t
indices = (0,0)


print("X: ")
print(X)

print("\nL: ")
print(matrixForwardSum(Xa = X, Wa = W, sigma = sigmoid))

print("\ndL_dX: ")
dLdX = matrixBackwardSum_X(Xa = X, Wa = W, sigma = sigmoid)
print(dLdX)
# First element 0.2489 is exactly like in the book: 
#assert dLdX[0,0] == Tensor([0.2489])
# TODO why doesn't this work????? 


# Testing whether the first element is the same as in dLdX
forwardIncr: FloatTensor = doForwardSumIncr_X(X, W, sigmoid, indices= indices)

print("\nforwardIncr: ")
print(forwardIncr)

# TODO why doesn't this work?
#assert torch.allclose(dLdX[indices], forwardIncr)




# %% [markdown]
# **Testing with 3-dim tensors:**
# 
# Can see that test is successful. The `dLdX[indices]` is approximately equal the `forwardIncr`:

# %% codecell
np.random.seed(190204)


X: Tensor = torch.arange(5*4*3).reshape(5,4,3)
W: Tensor = torch.rand(5,3,4)

indices = (0,1,0)


print("X: ")
print(X)

print("\nL: ")
print(matrixForwardSum(Xa = X, Wa = W, sigma = sigmoid))

print("\ndL_dX: ")
dLdX = matrixBackwardSum_X(Xa = X, Wa = W, sigma = sigmoid)
print(dLdX)

print("\ndLdX[{}]".format(indices))
print(dLdX[indices])
# First element 0.2489 is exactly like in the book: 
#assert dLdX[0,0] == Tensor([0.2489])
# TODO why doesn't this work????? 

# Testing whether the first element is the same as in dLdX
forwardIncr: FloatTensor = doForwardSumIncr_X(X, W, sigmoid, indices= indices)

print("\nforwardIncr: ")
print(forwardIncr)

assert torch.allclose(dLdX[indices], forwardIncr, rtol=0.01, atol=0.01)


# %% [markdown]
# **Testing with 4-dim tensors:**
# 
# Can see that test is successful. The `dLdX[indices]` is approximately equal the `forwardIncr`:

# %% codecell
np.random.seed(190204)


X: Tensor = torch.arange(5*4*3*2).reshape(5,2,4,3)
W: Tensor = torch.rand(5,2,3,1) #.type(torch.FloatTensor)


indices = (0,0,0,1)


print("X: ")
print(X)

print("\nL: ")
print(matrixForwardSum(Xa = X, Wa = W, sigma = sigmoid))

print("\ndL_dX: ")
dLdX = matrixBackwardSum_X(Xa = X, Wa = W, sigma = sigmoid)
print(dLdX)

print("\ndLdX[{}]".format(indices))
print(dLdX[indices])
# First element 0.2489 is exactly like in the book: 
#assert dLdX[0,0] == Tensor([0.2489])
# TODO why doesn't this work????? 

# Testing whether the first element is the same as in dLdX
forwardIncr: FloatTensor = doForwardSumIncr_X(X, W, sigmoid, indices= indices)

print("\nforwardIncr: ")
print(forwardIncr)

assert torch.allclose(dLdX[indices], forwardIncr, 
rtol=0.01, atol=0.01)

assert dLdX.shape == X.shape








# %% [markdown]
# ### Chain Rule Derivative in Code for $\frac{\partial L}{\partial W}$: 
#
# $$
# \begin{aligned}
# \frac{\partial L}{\partial W} &= \frac{\partial N}{\partial W} \times \bigg( \frac{\partial L}{\partial S} \odot \frac{\partial S}{\partial N} \bigg) \\
# &= X^T \times \bigg( \frac{\partial L}{\partial S} \odot \frac{\partial S}{\partial N} \bigg)
# \end{aligned}
# $$
# where $\odot$ signifies the Hadamard product and $\times$ is matrix multiplication.
# %% codecell


def matrixBackwardSum_W(Xa: Tensor, Wa: Tensor, sigma: TensorFunction) -> Tensor:
    '''Computes derivative of matrix function with respect to the first element X'''

    # NOTE: cast to float in case X or W is float else error   gets thrown
    X: Tensor = Xa.clone()
    W: Tensor = Wa.clone()
    # Now X and W are either float or long tensor

    TYPE_CAST = ['torch.FloatTensor', 'torch.DoubleTensor']

    if(Xa.type() in TYPE_CAST and Wa.type() == 'torch.LongTensor'):
         W = W.type(Xa.type())
    elif(Xa.type() == 'torch.LongTensor' and Wa.type() in TYPE_CAST):
         X = X.type(Wa.type())

    # Now X, W are either BOTH Long or BOTH Float tensors.


    isFirstPartEqualShape: bool = X.shape[:len(X.shape)-2] == W.shape[0:len(W.shape)-2]
    canDoMatMul: bool = X.shape[-1] == W.shape[-2]

    assert isFirstPartEqualShape and canDoMatMul

    ### Forward pass: Matrix multiplication:
    # X.shape = (...,n, m)
    # W.shape = (...,m, p)
    N: Tensor = torch.matmul(X, W)
    # N.shape = (...,n, p)

    # NOTE: result of matmul is either Float or Long Tensor depending on whether both X,W are Float or Long tensor. But they must be of the same kind (hence the if-else above)

    # Feeding the output of the matrix multiplication through sigma:
    S: Tensor = sigma(N)
    # S.shape = N.shape = (...,n, p)

    # Sum all the elements:
    #L: Tensor = torch.sum(S)
    # L.shape == (1 x 1)



    ### Backward pass: chain rule for matrix (df/dX) or (dS/dX)
    dL_dS: FloatTensor = torch.ones(S.shape)
    # dL_dS.shape == S.shape == (...,n, p)

    dS_dN = deriv(sigma, N) # NOTE: result is always a float tensor since deriv() is an approximation.
    # dS_dN.shape = N.shape = (...,n, p)
    dS_dN: FloatTensor = dS_dN.type(torch.FloatTensor) #  in case it is DoubleTensor so that no error when multiplying dlds and dsdn


    # NOTE: need to convert to float tensor because dS_dN is a float tensor, after the approx derivative calculation,
    # while W is just a long tensor and if we don't convert, we get runtime error. .
    dN_dW: FloatTensor = torch.transpose(X, X.ndim - 2, X.ndim - 1).type(torch.FloatTensor)
    # torch.transpose(W, 1, 0)
    ## dN_dW.shape = X^T.shape = (...,m,n)

     # Hadamard elementiwse product (*) followed by matrix multiplication (torch.matmul)
    dL_dW: FloatTensor = torch.matmul(dN_dW, (dL_dS * dS_dN) )
    #dL_dS * torch.matmul(dN_dW, dS_dN)
    # dL_dX.shape == (..., n,m)

    assert dL_dW.shape == W.shape

    return dL_dW ## shape == (...,n,m)



# %% [markdown]
# #### Testing if the derivatives computed are correct:
# A simple test is to perturb the array and observe the resulting change in output. If we increase $w_{i,j,k}$ by 0.001  we should see an increase in the value porduced by the forward function of the *gradient of the output with respect to $w_{i,j,k}$*.
# %% codecell

def doForwardSumIncr_W(Xa: Tensor, Wa: Tensor, sigma: TensorFunction, indices: Tuple[int]) -> Tensor:

     ## WARNING: the W must be FloatType tensors or else the later assertions here will fail! (only integer part of decimal gets copied)

     # NOTE: cast to float in case X or W is float else error   gets thrown
     X: Tensor = Xa.clone()
     W: Tensor = Wa.clone()
     # Now X and W are either float or long tensor

     TYPE_CAST = ['torch.FloatTensor', 'torch.DoubleTensor']

     if(Wa.type() in TYPE_CAST and Xa.type() == 'torch.LongTensor'):
          X = X.type(Wa.type())
     elif(Wa.type() == 'torch.LongTensor' and Xa.type() in TYPE_CAST):
          W = W.type(Xa.type())

     # Now X, W are either BOTH Long or BOTH Float tensors.
     #print("BEFORE: {}".format(W[indices]))
     
     Wclone = W.clone()

     # Creating the increment to be same size as the X[indices] shape: 
     increment = torch.tensor([0.001]).repeat_interleave(Wclone[indices].numel()).reshape(Wclone[indices].shape)

     # Increasing the value at that point by 0.01
     Wclone[indices] = W[indices] + increment

     #print("AFTER: {}".format(Wclone[indices]))
     #assert X[indices] == FLAG_NUM
     #assert Xclone[indices] == X[indices] + increment

     return ( (matrixForwardSum(X, Wclone, sigma) - matrixForwardSum(X, W, sigma)) / increment ).type(torch.FloatTensor)




# %% [markdown]
# **Testing with 2-dim tensors:**
# 
# Can see that test is successful. The `dLdX[indices]` is approximately equal the `forwardIncr`:
# %% codecell
np.random.seed(190204)


X: Tensor = Tensor(np.random.randn(3,3))
W: Tensor = Tensor(np.random.randn(3,2))

#sigma: TensorFunction = lambda t: 2*t + t
indices = (0,0)


print("W: ")
print(W)

print("\nL: ")
print(matrixForwardSum(Xa = X, Wa = W, sigma = sigmoid))

print("\ndL_dW: ")
dLdW = matrixBackwardSum_W(Xa = X, Wa = W, sigma = sigmoid)
print(dLdW)

#assert dLdW[0,0] == Tensor([0.2489])
# TODO why doesn't this work????? 


# Testing whether the first element is the same as in dLdX
forwardIncr: FloatTensor = doForwardSumIncr_W(X, W, sigmoid, indices= indices)

print("\nforwardIncr: ")
print(forwardIncr)


assert torch.allclose(dLdW[indices], forwardIncr, rtol=0.001, atol=0.001)




# %% [markdown]
# **Testing with 3-dim tensors:**
# 
# Can see that test is successful. The `dLdX[indices]` is approximately equal the `forwardIncr`:

# %% codecell
np.random.seed(190204)


X: Tensor = torch.arange(5*4*3).reshape(5,4,3)
W: Tensor = torch.rand(5,3,4)

indices = (0,1,0)


print("W: ")
print(W)

print("\nL: ")
print(matrixForwardSum(Xa = X, Wa = W, sigma = sigmoid))

print("\ndL_dW: ")
dLdW = matrixBackwardSum_W(Xa = X, Wa = W, sigma = sigmoid)
print(dLdW)

print("\ndLdW[{}]".format(indices))
print(dLdW[indices])

# Testing whether the first element is the same as in dLdX
forwardIncr: FloatTensor = doForwardSumIncr_W(X, W, sigmoid, indices= indices)

print("\nforwardIncr: ")
print(forwardIncr)

assert torch.allclose(dLdW[indices], forwardIncr, rtol=0.01, atol=0.01)


# %% [markdown]
# **Testing with 4-dim tensors:**
# 
# Can see that test is successful. The `dLdX[indices]` is approximately equal the `forwardIncr`:

# %% codecell
np.random.seed(190204)


X: Tensor = torch.arange(5*4*3*2).reshape(5,2,4,3)
W: Tensor = torch.rand(5,2,3,1) #.type(torch.FloatTensor)


indices = (0,0,2,0)


print("W: ")
print(W)

print("\nL: ")
print(matrixForwardSum(Xa = X, Wa = W, sigma = sigmoid))

print("\ndL_dW: ")
dLdW = matrixBackwardSum_W(Xa = X, Wa = W, sigma = sigmoid)
print(dLdW)

print("\ndLdW[{}]".format(indices))
print(dLdW[indices])
# First element 0.2489 is exactly like in the book: 
#assert dLdX[0,0] == Tensor([0.2489])
# TODO why doesn't this work????? 

# Testing whether the first element is the same as in dLdX
forwardIncr: FloatTensor = doForwardSumIncr_W(X, W, sigmoid, indices= indices)

print("\nforwardIncr: ")
print(forwardIncr)

assert torch.allclose(dLdW[indices], forwardIncr, rtol=0.01, atol=0.01)

assert dLdW.shape == W.shape






# %% codecell
