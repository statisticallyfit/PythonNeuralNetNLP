
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

# %% markdown [markdown]
# ## Derivative Function:
# $$
# \frac{df}{du}(a) = \lim_{\Delta \leftarrow 0} \frac{f(a + \Delta) - f(a - \Delta)}{2 \times \Delta}
# $$
# %% codecell

def deriv(func: TensorFunction, #Callable[[Tensor], Tensor],
     inputTensor: Tensor,
     delta: float = 0.001) -> Tensor:
     '''
     Evaluates the derivative of a function "func" at every element in the "inputTensor" array.
     '''
     return (func(inputTensor + delta) - func(inputTensor - delta)) / (2 * delta)

# %% markdown [markdown]
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



# %% markdown [markdown]
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




# %% markdown [markdown]
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





# %% markdown [markdown]
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






# %% markdown [markdown]
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

# %% markdown [markdown]
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




# %% markdown [markdown]
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






# %% markdown [markdown]
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

# %% markdown [markdown]
# ## Derivatives of Functions with Multiple Inputs (Case: Addition)
# Goal is to compute the derivative of each constituent function "going backward" through the computational graph and then multiply the result together to get the total derivative (as per chain rule).
# Given the function from before (calling it `f` now instead of `s`):
# $$
# f(x,y) = \sigma(\alpha(x,y)) = \sigma(x + y)
# $$
# %% markdown [markdown]
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
# %% markdown [markdown]
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
    #da_dx: Tensor = Tensor([1] * len(ds_da))

    df_dx: Tensor = ds_da * da_dx
    df_dy: Tensor = ds_da * da_dy

    return df_dx, df_dy

# %% codecell
x = Tensor(np.arange(-3, 8))
assert torch.equal(x, Tensor([-3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7]))


y = Tensor(np.arange(-5,6));  y
assert torch.equal(y, Tensor([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5]))

assert x.shape == y.shape

# Sigma is a cubing function
sigma: TensorFunction = lambda tensor: tensor**3

res: Tensor = multipleInputsAdd(x, y, sigma)
assert torch.equal(res , Tensor([-512, -216,  -64,   -8,  0,8,   64,  216,  512, 1000, 1728]))

# %% markdown [markdown]
# Printing out value of the derivatives with respect to $x$ and $y$:

# %% codecell
multipleInputsAddDeriv(x, y, sigma)







# %% markdown [markdown]
# ## Functions with Multiple Inputs (Case: Multiplication)
# Defining the following function $\alpha(x,y)$ with inputs $x$ and $y$:
# $$
# a = \beta(x, y) = x * y
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

# %% markdown [markdown]
# ## Derivatives of Functions with Multiple Inputs (Case: Multiplication)
# Goal is to compute the derivative of each constituent function "going backward" through the computational graph and then multiply the result together to get the total derivative (as per chain rule).
# Given the function from before (calling it `f` now instead of `s`):
# $$
# f(x,y) = \sigma(\beta(x,y)) = \sigma(x * y)
# $$
# %% markdown [markdown]
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
# %% markdown [markdown]
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
x = Tensor(np.arange(-3, 8))
assert torch.equal(x, Tensor([-3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7]))


y = Tensor(np.arange(-5,6));  y
assert torch.equal(y, Tensor([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5]))

assert x.shape == y.shape

# Sigma is a cubing function
sigma: TensorFunction = lambda tensor: tensor**3

# %% codecell
beta: Tensor = x*y; beta
# %% codecell
deriv(sigma, beta)
# %% codecell
multipleInputsMultiply(x, y, sigma)



# %% markdown [markdown]
# ## Functions with Multiple Matrix Inputs
# Here, inputs are multi-dimensional tensors, not just 1-dim tensors as in the above two examples.
#
# Define the following:
# $$
# X = \Big[ x_1 \;\; x_2 \;\; ... \;\; x_n \Big]
# $$
#
# $$
# \begin{align}
#   W &=\begin{bmatrix}
#       w_1 \\
#       w_2 \\
#       \vdots \\
#       w_n
#   \end{bmatrix}
# \end{align}
# $$
# where $x_i, w_i \in \mathbf{R}^n$ so the elements of $X$ and $W$ themselves can also be tensors.
#
# Define a function to carry out the tensor multiplication between these tensors:
# $$
# \begin{align}
# N &= \nu(X, W) \\
#   &= X \times W \\
#   &= \Big[ x_1 \;\; x_2 \;\; ... \;\; x_n \Big] \times \begin{bmatrix}  w_1 \\ w_2 \\  \vdots \\  w_n \end{bmatrix} \\
#   &= x_1 \times w_1 + x_2 \times w_2 + ... + x_n \times w_n
# \end{align}
# $$

# %% codecell
def matmulForward(X: Tensor, W: Tensor) -> Tensor:
    '''Computes the forward pass of a matrix multiplication'''

    assert X.shape[1] == W.shape[0], \
        '''For matrix multiplication, with X.shape == (m, n) and W.shape == (r, p), it is required that n == r. Instead, the number of columns in the first tensor X is n == {0} and the number of rows in the second tensor is r == {1}'''.format(X.shape[1], W.shape[0])

    # matrix multiplication
    #N: Tensor = np.dot(X, W) # same as torch.matmul
    N: Tensor = torch.matmul(X, W)

    assert torch.equal(Tensor(np.dot(X, W)), N)

    return N

# NOTE: difference between np.matmul and np.dot:
# * np.matmul: https://hyp.is/YXQQAPJoEeq_U1-RXfLGDQ/numpy.org/doc/stable/reference/generated/numpy.matmul.html
# * np.dot: https://hyp.is/S3p4evJoEeqQlvsyite7fg/numpy.org/doc/stable/reference/generated/numpy.dot.html
#
#
# %% markdown [markdown]
# ## Derivatives of Functions with Multiple Tensor Inputs
