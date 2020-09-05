
# %% codecell
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import ndarray
%matplotlib inline
from typing import *

import torch
import torch.tensor as Tensor


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
# A function takes in an array (Tensor) as an argument and produces another Tensor.
TensorFunction = Callable[[Tensor], Tensor]

# A Chain is a list of functions:
Chain = List[TensorFunction]

# Define how data goes through a chain for a list of length 2:
def chainLength2(chain: Chain, array: Tensor) -> Tensor:
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

def square(x: Tensor) -> Tensor:
     return Tensor(np.power(x, 2))

def leakyRelu(x: Tensor) -> Tensor:
     '''Apply leaky relu function to each element in the tensor'''
     return Tensor(np.maximum(0.2 * x, x))


def sigmoid(x: Tensor) -> Tensor:
     ''' Apply the sigmoid function to each element in the input array (Tensor)'''

     return Tensor(1 / (1 + np.exp(-x)))


# %% codecell
# Chain rule for two composed functions:
def chainDeriv2(chain: Chain,  inputRange: Tensor) -> Tensor:
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
PLOT_RANGE = np.arange(-3, 3, 0.01)

chain1: Chain = [square, sigmoid]
