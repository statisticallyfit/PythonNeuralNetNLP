
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

# %% markdown [markdown]
# Testing the abstract functions chainFunction and chainDeriv:
# %% codecell
x: Tensor = Tensor(np.arange(-3, 8)); x
chain: List[TensorFunction] = [leakyRelu, sigmoid, square, cubic, quartic, quintic, sinT, cosT]

# %% codecell

assert torch.equal(chainDeriv_OneInputTensorFunc(chain[0:3], x), chainDerivThree(chain[0:3], x))
assert torch.equal(chainFunctions(chain[0:3], x), chainThreeFunctions(chain[0:3], x))

# %% codecell
chainDeriv_OneInputTensorFunc(chain[0:3], x)
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

     outputRange: Tensor = chainDeriv_OneInputTensorFunc(chain = chain, X= inputRange)

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

    assert torch.equal(Tensor(np.dot(X, W)), N)

    return N

# %% markdown [markdown]
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

# %% markdown [markdown]
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
# %% markdown
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
assert torch.equal(Tensor(dN_dX.numpy().transpose((1,0,2))), torch.transpose(dN_dX,1,0))


# %% markdown
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



# %% markdown
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
#   &= \frac{\partial \sigma}{\partial \nu} \bigg|_{\large \nu = x_1 \times w_1 + ... + x_n \times w_n} \times \frac{\partial \nu}{\partial X}  \\
#   &= \frac{\partial \sigma}{\partial \nu}\bigg|_{\large \nu = x_1 \times w_1 + ... + x_n \times w_n} \times W^T \\
#   &= \frac{\partial}{\partial \nu}(\sigma(\nu(X,W))) \bigg|_{\large \nu = x_1 \times w_1 + ... + x_n \times w_n} \times W^T \\
#   &= \frac{\partial}{\partial \nu}(\sigma(x_1 \times w_1 + ... + x_n \times w_n)) \times W^T
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
# %% markdown
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


# %% markdown
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
# %% markdown
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


# %% markdown
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


## Computation Graph with 2D Matrix Inputs
# %% markdown
# ## Functions with Multiple Matrix Inputs: (Forward Pass) Lambda Sum
# Define the following, where $X$ and $W$ are now 2D matrices instead of higher-dim tensors (letting $X$ be $n \times m$ and $W$ be $m \times p$):
# $$
# \begin{align}
# X &= \begin{pmatrix}
# x_{11} & x_{12} & ... & x_{1m} \\
# x_{21} & x_{22} & ... & x_{2m} \\
# \vdots & \vdots & ... & \vdots \\
# x_{n1} & x_{n2} & ... & x_{nm}
# \end{pmatrix}
# \end{align}
# $$
#
# $$
# \begin{align}
# W &= \begin{pmatrix}
# w_{11} & w_{12} & ... & w_{1p} \\
# w_{21} & w_{22} & ... & w_{2p} \\
# \vdots & \vdots & ... & \vdots \\
# w_{m1} & w_{m2} & ... & w_{mp}
# \end{pmatrix}
# \end{align}
# $$
# %% markdown
# Define some straightforward operations on these matrices:
#
# ### STEP 1:
# Multiply these matrices together using $N = \nu(X, W)$ and denoting the row $i$ and column $j$ in the resulting matrix as $(XW)_{ij}$:
#
# $$
# \begin{align}
# N &= \nu(X,W) \\
# &= X \times W \\
# &= \begin{pmatrix}
#   x_{11} \cdot w_{11} + x_{12} \cdot w_{21} + ... + x_{1m} \cdot w_{m1} &
#   x_{11} \cdot w_{12} + x_{12} \cdot w_{22} + ... + x_{1m} \cdot w_{m2} &
#   ... &
#   x_{11} \cdot w_{1p} + x_{12} \cdot w_{2p} + ... + x_{1m} \cdot w_{mp} \\
#
#   x_{21} \cdot w_{11} + x_{22} \cdot w_{21} + ... + x_{2m} \cdot w_{m1} &
#   x_{21} \cdot w_{12} + x_{22} \cdot w_{22} + ... + x_{2m} \cdot w_{m2} &
#   ... &
#   x_{21} \cdot w_{1p} + x_{22} \cdot w_{2p} + ... + x_{2m} \cdot w_{mp} \\
#
#   \vdots & \vdots & \vdots & \vdots \\
#
#   x_{n1} \cdot w_{11} + x_{n2} \cdot w_{21} + ... + x_{nm} \cdot w_{m1} &
#   x_{n1} \cdot w_{12} + x_{n2} \cdot w_{22} + ... + x_{nm} \cdot w_{m2} &
#   ... &
#   x_{n1} \cdot w_{1p} + x_{n2} \cdot w_{2p} + ... + x_{nm} \cdot w_{mp} \\
# \end{pmatrix} \\
#
# &= \begin{pmatrix}
#   (XW)_{11} &   (XW)_{12} & ... & (XW)_{1p} \\
#   (XW)_{21} &   (XW)_{22} & ... & (XW)_{2p} \\
#   \vdots & \vdots & \vdots & \vdots \\
#   (XW)_{n1} &   (XW)_{n2} & ... & (XW)_{np}
# \end{pmatrix}
# \end{align}
# $$
# where `X.shape == (n, m)`, and `W.shape == (m, p)`, and `N.shape == (n, p)`.
#
# ### STEP 2:
# Feed $N$ through some differentiable function $\sigma$ and define $S = \sigma(N)$ (just applying $\sigma$ to every element of the matrix operation defined by $N$):
#
# $$
# \begin{align}
# S &= \sigma(N) \\
#  &= \sigma(\nu(X, W)) \\
#  &= \sigma(X \times W) \\
#  &= \begin{pmatrix}
#   \sigma \Big( x_{11} \cdot w_{11} + x_{12} \cdot w_{21} + ... + x_{1m} \cdot w_{m1} \Big) &
#   \sigma \Big( x_{11} \cdot w_{12} + x_{12} \cdot w_{22} + ... + x_{1m} \cdot w_{m2} \Big) &
#   ... &
#   \sigma \Big( x_{11} \cdot w_{1p} + x_{12} \cdot w_{2p} + ... + x_{1m} \cdot w_{mp} \Big) \\
#
#   \sigma \Big( x_{21} \cdot w_{11} + x_{22} \cdot w_{21} + ... + x_{2m} \cdot w_{m1} \Big) &
#   \sigma \Big( x_{21} \cdot w_{12} + x_{22} \cdot w_{22} + ... + x_{2m} \cdot w_{m2} \Big) &
#   ... &
#   \sigma \Big( x_{21} \cdot w_{1p} + x_{22} \cdot w_{2p} + ... + x_{2m} \cdot w_{mp} \Big) \\
#
#   \vdots & \vdots & \vdots & \vdots \\
#
#   \sigma \Big( x_{n1} \cdot w_{11} + x_{n2} \cdot w_{21} + ... + x_{nm} \cdot w_{m1} \Big) &
#   \sigma \Big( x_{n1} \cdot w_{12} + x_{n2} \cdot w_{22} + ... + x_{nm} \cdot w_{m2} \Big) &
#   ... &
#   \sigma \Big( x_{n1} \cdot w_{1p} + x_{n2} \cdot w_{2p} + ... + x_{nm} \cdot w_{mp} \Big) \\
# \end{pmatrix} \\
#
# &= \begin{pmatrix}
#   \sigma \Big( (XW)_{11} \Big) &   \sigma \Big( (XW)_{12} \Big) & ... & \sigma \Big( (XW)_{1p} \Big) \\
#   \sigma \Big( (XW)_{21} \Big) &   \sigma \Big( (XW)_{22} \Big) & ... & \sigma \Big( (XW)_{2p} \Big) \\
#   \vdots & \vdots & \vdots & \vdots \\
#   \sigma \Big( (XW)_{n1} \Big) &   \sigma \Big( (XW)_{n2} \Big) & ... & \sigma \Big( (XW)_{np} \Big)
# \end{pmatrix}
# \end{align}
# $$
# where `S.shape == N.shape == (n, p)`.
#
# ### STEP 3:
# Defining a $\Lambda$ function to sum up the elements in the matrix to find the total effect of changing each element of a matrix:
#
# $$
# \begin{align}
# L &= \Lambda(\sigma(N(X, W))) \\
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
# \end{align}
# $$
#  where $\Lambda$.`shape == (1,1)` so $\Lambda$ is a constant
#
# * NOTE: the code can handle higher-dim tensors while the formulas above handle only 2-dim tensors (matrices). So in the code, we have:
#   * `X.shape == (..., n, m)`
#   * `W.shape == (..., m, p)`
#   * `N.shape == (..., n, p)`
#   * `S.shape == (..., n, p)`
#   * $\Lambda$`.shape == (1,1)`
# * and in the formulas we have:
#   * `X.shape == (n, m)`
#   * `W.shape == (m, p)`
#   * `N.shape == (n, p)`
#   * `S.shape == (n, p)`
#   * $\Lambda$`.shape == (1,1)`

# * NOTE: the '...' is set to mean that all dimensions before the last two are the same for all tensors in these calculations.
# %% codecell
def matrixForwardSum(X: Tensor, W: Tensor, sigma: TensorFunction) -> float:
     '''Computes the result of the forward pass of the function L with input tensors X and W and function sigma

     X.shape == (..., n, m)
     W.shape == (..., m, p)
     '''

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



# %% markdown
# ## Derivative of Functions with Multiple Matrix Inputs: (Backward Pass) Lambda Sum

# ### Background: Jacobian Matrix and Multivariable Functions
# A vector $\mathbf{f} = \big( f_1, f_2, ..., f_m \big)$ of $m$ functions, each depending on $n$ variables $\mathbf{x} = \big(x_1, x_2, ..., x_n \big)$ defines a transformation or function from $\mathbb{R}^n$ to $\mathbb{R}^m$. Specifically, if $\mathbf{x} \in \mathbb{R}^n$ and if:
# $$
# y_1 = f_1 \big(x_1,x_2,...,x_n \big) \\
# y_2 = f_2 \big(x_1,x_2,...,x_n \big) \\
# \vdots \\
# y_m = f_m \big(x_1,x_2,...,x_n \big)
# $$
# then $\mathbf{y} = \big(y_1, y_2, ..., y_m \big)$ is the point in $\mathbb{R}^m$ that corresponds to $\mathbf{x}$ under the transformation $\mathbf{f}$. We can write these equations more compactly as:
# $$
# \mathbf{y} = \mathbf{f}(\mathbf{x})
# $$
# Information about the rate of change of $\mathbf{y}$ with respect to $\mathbf{x}$ is contained in the various partial derivatives $\frac{\partial y_i}{\partial x_j}$ for $1 \leq i \leq m, 1 \leq j \leq n$ and is conveniently organized into an $m \times n$ matrix $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ called the **Jacobian matrix** of the transformation $\mathbf{f}$. The Jacobian matrix is the collection of all $m \times n$ possible partial derivatives ($m$ rows and $n$ columns), which is the stack of $m$ gradients with respect to $\mathbf{x}$:
# $$
# \Large
# \begin{align}
# \frac{\partial \mathbf{y}}{\partial \mathbf{x}} &= \begin{pmatrix}
#    \nabla f_1(\mathbf{x}) \\
#    \nabla f_2(\mathbf{x}) \\
#    \vdots \\
#    \nabla f_m(\mathbf{x})
# \end{pmatrix}
#
# = \begin{pmatrix}
#    \frac{\partial}{\partial \mathbf{x}} f_1(\mathbf{x}) \\
#    \frac{\partial}{\partial \mathbf{x}} f_2(\mathbf{x}) \\
#    \vdots \\
#    \frac{\partial}{\partial \mathbf{x}} f_m(\mathbf{x})
# \end{pmatrix}
#
# = \begin{pmatrix}
#   \frac{\partial}{\partial x_1} f_1(\mathbf{x}) & \frac{\partial}{\partial x_2} f_1(\mathbf{x}) & ... & \frac{\partial}{\partial x_n} f_1(\mathbf{x}) \\
#   \frac{\partial}{\partial x_1} f_2(\mathbf{x}) & \frac{\partial}{\partial x_2} f_2(\mathbf{x}) & ... & \frac{\partial}{\partial x_n} f_2(\mathbf{x}) \\
#   \vdots & \vdots &  & \vdots \\
#   \frac{\partial}{\partial x_1} f_m(\mathbf{x}) & \frac{\partial}{\partial x_2} f_m(\mathbf{x}) & ... & \frac{\partial}{\partial x_n} f_m(\mathbf{x})
# \end{pmatrix} \\
#
# \frac{\partial \mathbf{y}}{\partial \mathbf{x}} &= \begin{pmatrix}
#   \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & ... & \frac{\partial f_1}{\partial x_n} \\
#   \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & ... & \frac{\partial f_2}{\partial x_n} \\
#   \vdots & \vdots &  & \vdots \\
#   \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & ... & \frac{\partial f_m}{\partial x_n}
# \end{pmatrix}
# \end{align}
# $$
# This linear transformation represented by the Jacobian matrix is called **the derivative** of the transformation $\mathbf{f}$.
#
# Each $\frac{\partial f_i}{\partial \mathbf{x}} is a horizontal $n$-vector because the partial derivative is with respect to a vector $\mathbf{x}$ whose length is $n = |\mathbf{x}|$, making the width of the Jacobian $n$ (there are $n$ parameters that are variable, each potentially changing the function's value).
#

# %% markdown
# ### Background: Chain Rule for Matrices
# In general the Jacobian matrix of the composition of two vector-valued functions of a vector variable is the matrix product of their Jacobian matrices.
#
# To see this let $\mathbf{z} = \mathbf{f}(\mathbf{y})$ be a transformation from $\mathbb{R}^k$ to $\mathbb{R}^m$ given by:
# $$
# z_1 = f_1 \big(y_1,y_2,...,y_k \big) \\
# z_2 = f_2 \big(y_1,y_2,...,y_k \big) \\
# \vdots \\
# z_k = f_m \big(y_1,y_2,...,y_k \big)
# $$
# which has the $m \times k$ Jacobian matrix:
# $$
# \Large
# \frac{\partial \mathbf{f}}{\partial \mathbf{y}} = \begin{pmatrix}
#   \frac{\partial f_1}{\partial y_1} & \frac{\partial f_1}{\partial y_2} & ... & \frac{\partial f_1}{\partial y_k} \\
#   \frac{\partial f_2}{\partial y_1} & \frac{\partial f_2}{\partial y_2} & ... & \frac{\partial f_2}{\partial y_k} \\
#   \vdots & \vdots &  & \vdots \\
#   \frac{\partial f_m}{\partial y_1} & \frac{\partial f_m}{\partial y_2} & ... & \frac{\partial f_m}{\partial y_k}
# \end{pmatrix}
# $$
#

# and let $\mathbf{y} = \mathbf{g}(\mathbf{x})$ be another such transformation from $\mathbb{R}^n$ to $\mathbb{R}^k$ given by:
#
# $$
# y_1 = g_1 \big(x_1,x_2,...,x_n \big) \\
# y_2 = g_2 \big(x_1,x_2,...,x_n \big) \\
# \vdots \\
# y_k = g_k \big(x_1,x_2,...,x_n \big)
# $$
# which has the $k \times n$ Jacobian matrix:
# $$
# \Large
# \frac{\partial \mathbf{g}}{\partial \mathbf{x}} = \begin{pmatrix}
#   \frac{\partial g_1}{\partial x_1} & \frac{\partial g_1}{\partial x_2} & ... & \frac{\partial g_1}{\partial x_n} \\
#   \frac{\partial g_2}{\partial x_1} & \frac{\partial g_2}{\partial x_2} & ... & \frac{\partial g_2}{\partial x_n} \\
#   \vdots & \vdots &  & \vdots \\
#   \frac{\partial g_k}{\partial x_1} & \frac{\partial g_k}{\partial x_2} & ... & \frac{\partial g_k}{\partial x_n}
# \end{pmatrix}
# $$
#
# Then the composition $\mathbf{z} = (\mathbf{f} \circ \mathbf{g})(\mathbf{x}) = \mathbf{f}(\mathbf{g}(\mathbf{x}))$ given by :
# $$
# z_1 = f_1 \big( g_1 \big( x_1,...,x_n \big),..., g_k \big( x_1,...,x_n \big) \big) \\
# z_2 = f_2 \big( g_1 \big( x_1,...,x_n \big),..., g_k \big( x_1,...,x_n \big) \big) \\
# \vdots \\
# z_k = f_m \big( g_1 \big( x_1,...,x_n \big),..., g_k \big( x_1,...,x_n \big) \big)
# $$
#
# has, according to the Chain Rule, the $m \times n$ Jacobian matrix
#
# $$
# \Large
# \begin{align}
#
# \frac{\partial}{\partial \mathbf{x}} \mathbf{f} \big( \mathbf{g}(\mathbf{x}) \big) &= \frac{\partial \mathbf{f}}{\partial \mathbf{g}} \times \frac{\partial \mathbf{g}}{\partial \mathbf{x}} \\
#
# \begin{pmatrix}
# \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & ... & \frac{\partial f_1}{\partial x_n} \\
# \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & ... & \frac{\partial f_2}{\partial x_n} \\
# \vdots & \vdots & & \vdots \\
# \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & ... & \frac{\partial f_m}{\partial x_n}
# \end{pmatrix}
#
# &= \begin{pmatrix}
# \frac{\partial f_1}{\partial g_1} & \frac{\partial f_1}{\partial g_2} & ... & \frac{\partial f_1}{\partial g_k} \\
# \frac{\partial f_2}{\partial g_1} & \frac{\partial f_2}{\partial g_2} & ... & \frac{\partial f_2}{\partial g_k} \\
# \vdots & \vdots & & \vdots \\
# \frac{\partial f_m}{\partial g_1} & \frac{\partial f_m}{\partial g_2} & ... & \frac{\partial f_m}{\partial g_k}
# \end{pmatrix}
#
# \times
#
# \begin{pmatrix}
# \frac{\partial g_1}{\partial x_1} & \frac{\partial g_1}{\partial x_2} & ... & \frac{\partial g_1}{\partial x_n} \\
# \frac{\partial g_2}{\partial x_1} & \frac{\partial g_2}{\partial x_2} & ... & \frac{\partial g_2}{\partial x_n} \\
# \vdots & \vdots & & \vdots \\
# \frac{\partial g_k}{\partial x_1} & \frac{\partial g_k}{\partial x_2} & ... & \frac{\partial g_k}{\partial x_n}
# \end{pmatrix}
#
# \end{align}
# $$
# where $\times$ denotes matrix multiplication, and $m = |\mathbf{f}|, n = |\mathbf{x}|$ and $k = |\mathbf{g}|$.
#
#
# **SOURCES:**
# * R.A Adams - Calculus: A Complete Course (sections 12.5 and 12.6)
# * Thomas Weir - Calculus (section 14.4)
# * [Medium's blog post on "The Matrix Calculus you Need for Deep Learning"](https://medium.com/@rohitrpatil/the-matrix-calculus-you-need-for-deep-learning-notes-from-a-paper-by-terence-parr-and-jeremy-4f4263b7bb8)






# %% markdown
# ### Background: Chain Rule for Matrices (from Adams)
# In general the Jacobian matrix of the composition of two vector-valued functions of a vector variable is the matrix product of their Jacobian matrices.
#
# To see this let $\mathbf{y} = \mathbf{f}(\mathbf{x})$ be a transformation from $\mathbb{R}^n$ to $\mathbb{R}^m$ as above and let $\mathbf{z} = \mathbf{g}(\mathbf{y})$ be another such transformation from $\mathbb{R}^m$ to $\mathbb{R}^k$ given by:
# $$
# z_1 = g_1 \big(y_1,y_2,...,y_m \big) \\
# z_2 = g_2 \big(y_1,y_2,...,y_m \big) \\
# \vdots \\
# z_k = g_k \big(y_1,y_2,...,y_m \big)
# $$
# which has the $k \times m$ Jacobian matrix:
# $$
# \begin{align}
# \Large
# \frac{\partial \mathbf{g}}{\partial \mathbf{y}} = \begin{pmatrix}
#   \frac{\partial z_1}{\partial y_1} & \frac{\partial z_1}{\partial y_2} & ... & \frac{\partial z_1}{\partial y_m} \\
#   \frac{\partial z_2}{\partial y_1} & \frac{\partial z_2}{\partial y_2} & ... & \frac{\partial z_2}{\partial y_m} \\
#   \vdots & \vdots &  & \vdots \\
#   \frac{\partial z_k}{\partial y_1} & \frac{\partial z_k}{\partial y_2} & ... & \frac{\partial z_k}{\partial y_m}
# \end{pmatrix}
# \end{align}
# $$
# Then the composition $\mathbf{z} = (\mathbf{g} \circ \mathbf{f})(\mathbf{x}) = \mathbf{g}(\mathbf{f}(\mathbf{x}))$ given by :
# $$
# z_1 = g_1 \big( f_1 \big( x_1,...,x_n \big),..., f_m \big( x_1,...,x_n \big) \big) \\
# z_2 = g_2 \big( f_1 \big( x_1,...,x_n \big),..., f_m \big( x_1,...,x_n \big) \big) \\
# \vdots \\
# z_k = g_k \big( f_1 \big( x_1,...,x_n \big),..., f_m \big( x_1,...,x_n \big) \big)
# $$
#
# has, according to the Chain Rule, the $k \times n$ Jacobian matrix
#
# $$
# \Large
# \begin{align}
#
# \frac{\partial}{\partial \mathbf{x}} \mathbf{g} \big( \mathbf{f}(\mathbf{x}) \big) &= \frac{\partial \mathbf{g}}{\partial \mathbf{f}} \times \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \\
#
#
# \begin{pmatrix}
# \frac{\partial z_1}{\partial x_1} & \frac{\partial z_1}{\partial x_2} & ... & \frac{\partial z_1}{\partial x_n} \\
# \frac{\partial z_2}{\partial x_1} & \frac{\partial z_2}{\partial x_2} & ... & \frac{\partial z_2}{\partial x_n} \\
# \vdots & \vdots & & \vdots \\
# \frac{\partial z_k}{\partial x_1} & \frac{\partial z_k}{\partial x_2} & ... & \frac{\partial z_k}{\partial x_n}
# \end{pmatrix}
#
# &= \begin{pmatrix}
# \frac{\partial z_1}{\partial y_1} & \frac{\partial z_1}{\partial y_2} & ... & \frac{\partial z_1}{\partial y_m} \\
# \frac{\partial z_2}{\partial y_1} & \frac{\partial z_2}{\partial y_2} & ... & \frac{\partial z_2}{\partial y_m} \\
# \vdots & \vdots & & \vdots \\
# \frac{\partial z_k}{\partial y_1} & \frac{\partial z_k}{\partial y_2} & ... & \frac{\partial z_k}{\partial y_m}
# \end{pmatrix}
#
# \times
#
# \begin{pmatrix}
# \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & ... & \frac{\partial y_1}{\partial x_n} \\
# \frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & ... & \frac{\partial y_2}{\partial x_n} \\
# \vdots & \vdots & & \vdots \\
# \frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & ... & \frac{\partial y_m}{\partial x_n}
# \end{pmatrix}
#
# \end{align}
# $$
# where $\times$ denotes matrix multiplication.
#
#
# **SOURCES:**
# * R.A Adams - Calculus: A Complete Course (sections 12.5 and 12.6)
# * Thomas Weir - Calculus (section 14.4)
# * [Medium's blog post on "The Matrix Calculus you Need for Deep Learning"](https://medium.com/@rohitrpatil/the-matrix-calculus-you-need-for-deep-learning-notes-from-a-paper-by-terence-parr-and-jeremy-4f4263b7bb8)
# %% markdown
# -----------------------------------------------------------------------------------
# We have a number $L$ and we want to find out the gradient of $L$ with respect to $X$ and $W$; how much changing *each element* of these input matrices (so each $x_{ij}$ and each $w_{ij}$) would change $L$.
#
# **Direct way:**
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
# **Chain Rule way (Backward pass):**
# $$
# \frac{\partial \Lambda}{\partial X} = \frac{\partial N}{\}
# $$

# TODO: continue from medium article + waterloo bit (tabs recorded in snagit) to explain starting from Jacobian concept.




# %% codecell

# TODO: equivalent for the matrix W (understand why dot first then matmul in the chain rule)
def matrixBackwardSum_X(X: Tensor, W: Tensor, sigma: TensorFunction) -> Tensor:
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

    # Sum all the elements:
    L: Tensor = torch.sum(S)
    # L.shape == (1 x 1)



    ### Backward pass: chain rule for matrix (df/dX) or (dS/dX)
    dL_dS: Tensor = torch.ones(S.shape)
    # dL_dS.shape == S.shape == (...,n, p)

    dS_dN: Tensor = deriv(sigma, N)
    # dS_dN.shape = N.shape = (...,n, p)

    # TODO FIGURE OUT why we have element-wise multiplication here:
    dL_dN: Tensor = dL_dS * dS_dN
    # dL_dN.shape == (...,n, p)

    # Transpose along first two dimensions (this example assumes we are
    # using 2-dim tensors which are matrices)

    # NOTE: need to convert to float tensor because dS_dN is a float tensor, after the approx derivative calculation,
    # while W is just a long tensor and if we don't convert, we get runtime error. .
    dN_dX: Tensor = torch.transpose(W, W.ndim - 2, W.ndim - 1).type(torch.FloatTensor)
    # torch.transpose(W, 1, 0)
    ## dN_dX.shape = W^T.shape = (...,p,m)


    # TODO: why matrix multiplication here and why element wise multiply above?
    dL_dX: Tensor = torch.matmul(dS_dN, dN_dX)
    # dL_dX.shape == (..., n,m)

    assert dL_dX.shape == X.shape

    return dL_dX ## shape == (...,n,m)


# %% codecell

####### TODO here -------------------------------------------------------------

X: Tensor = torch.arange(3*4*2*7).reshape(3,2,4,7)
W: Tensor = torch.arange(3*5*2*7).reshape(3,2,7,5)
sigma: TensorFunction = lambda t: t**3 + t


assert matrixBackwardSigma_X(X, W, sigma).shape == (3, 2, 4, 7)

# %% codecell
x: Tensor = torch.rand(2,10)
w: Tensor = torch.rand(10,2)

matrixBackwardSigma_X(x, w, sigmoid)
# %% markdown
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


# %% markdown
# Testing with 2-dim tensors:
# %% codecell
X: Tensor = torch.arange(5*4).reshape(5,4).type(torch.FloatTensor)
W: Tensor = torch.rand(4,5)

indices = (2,1)
increment = 0.01
inc: Tensor = doForwardSigmaIncr(X, W, sigma, indices = indices, increment = increment)
incNot: Tensor = doForwardSigmaIncr(X, W, sigma, indices = indices, increment = 0)


print(((inc - incNot)/increment).sum())

print(matrixBackwardSigma_X(X, W, sigma)[indices])
# %% markdown
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
#matrixBackwardExtra_X(X, W, sigma)[2,1,3]
# %% markdown
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
