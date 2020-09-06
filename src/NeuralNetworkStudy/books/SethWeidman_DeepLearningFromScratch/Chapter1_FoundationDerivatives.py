
# %% codecell
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import ndarray
%matplotlib inline
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

sys.path.append(NEURALNET_PATH)
assert NEURALNET_PATH in sys.path


from FunctionUtil import *
from TypeUtil import *

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
# \frac {d} {dx} (g(f(x))) = \frac {dg} {df} \frac {df}{dx}
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

def plotChain(ax, chain: Chain, inputRange: Tensor) -> None:
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


     outputRange: Tensor = chainTwoFunctions(chain = chain, x = inputRange)
     ax.plot(inputRange, outputRange)



def plotChainDeriv(ax, chain: Chain, inputRange: Tensor) -> None:
     """Uses the chain rule to plot the derivative of a function consisting of two nested functions.

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
     outputRange: Tensor = chainDerivTwo(chain = chain, inputRange = inputRange)
     ax.plot(inputRange, outputRange)






# %% codecell
PLOT_RANGE: Tensor = Tensor(np.arange(-3, 3, 0.01))

chain1: Chain = [square, sigmoid]
chain2: Chain = [sigmoid, square]

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 8)) # 2 rows, 1 column

# First chain (first nesting is sigmoid inner, square outer)
plotChain(ax = ax[0], chain = chain1, inputRange = PLOT_RANGE)
plotChainDeriv(ax = ax[0], chain = chain1, inputRange = PLOT_RANGE)

ax[0].legend(["$f(x)$", "$\\frac{df}{dx}$"])
ax[0].set_title("Function and derivative for\n$f(x) = sigmoid(square(x))$")

# Second chain (second nesting is square inner, sigmoid outer)
plotChain(ax = ax[1], chain = chain2, inputRange = PLOT_RANGE)
plotChainDeriv(ax = ax[1], chain = chain2, inputRange = PLOT_RANGE)
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
# \frac{d}{dx}(h(g(f(x)))) = \frac{dh}{d(g \circ f)} \times \frac {}
# $$
#
# Prime notation of chain rule:
# $$
#
# $$


# %% markdown
# Leibniz notation:
# $$
# \frac {d} {dx} (g(f(x))) = \frac {dg} {df} \frac {df}{dx}
# $$
# Prime notation:
# $$
# (g(f(x)))' = g'(f(x)) \cdot f'(x)
# $$
