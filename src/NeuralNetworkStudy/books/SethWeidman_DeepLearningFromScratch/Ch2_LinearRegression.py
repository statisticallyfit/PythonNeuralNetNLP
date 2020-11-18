# %% [markdown]
# # Chapter 2: Linear Regression with Neural Networks
#
# Chapter 2 of Seth Weidman's Deep Learning from Scratch - Building with Python from First Principles.
#
# **My Enrichment:** ___

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


# %% markdown
# ## Linear Regression Formulas
#
# ### Data Representation
#
# The data is represented in a matrix $X$ with $n$ rows, each of which represents an observation with $k$ features, all of which are numbers. Each row observation is a vector $\overrightarrow{x}_i = \begin{pmatrix}x_{i1}  &  x_{i2}  &  x_{i3} & ... & x_{ik} \end{pmatrix}$. Each of these row observation vectors will be stacked on top of one another to form a $b \times k$ batch matrix, $X_{\text{batch}}$.
#
# For instance a batch of size $b$ would look like this:
# $$
# \begin{aligned}
# X_\text{batch} = \begin{pmatrix}
#   x_{11} & x_{12} & ... & x_{1k} \\
#   x_{21} & x_{22} & ... & x_{2k} \\
#   \vdots & \vdots & ... & \vdots \\
#   x_{b1} & x_{b2} & ... & x_{bk}
# \end{pmatrix}
# = \begin{pmatrix}
# \overrightarrow{x}_1 \\ \overrightarrow{x}_2 \\ \vdots \\ \overrightarrow{x}_b
# \end{pmatrix}
# \end{aligned}
# $$
# where ...
# * $k$ = number of data features or predictors
# * $n$ = number of rows or number of data observations
# * $b$ = number of batches
#
#
# $\color{red}{\text{TODO question: In the entire data matrix } X \text{ there can be many batches } b \text{ fit inside the } n \text{ rows. Does it make sense to think of } X \text{ as at least a 3-dimensional tensor ?? }}$
#
#
# ### Target Representation
#
# For each batch of observations $X_{\text{batch}}$ there is a corresponding batch of *targets*, each element of which is the target number for the corresponding observation. For instance $\overrightarrow{y}_i$ is the target number for observation vector $\overrightarrow{x}_i$.
#
# We can represent that batch of targets for $X_\text{batch}$ in a one-dimensional vector, $b \times 1$ vector:
#
# $$
# \overrightarrow{y}_\text{batch} = \begin{pmatrix}
#   y_1 \\
#   y_2 \\
#   \vdots \\
#   y_b
# \end{pmatrix}
# $$
# where $b$ = batch size.
#
#
# ## Linear Regression Formulation
#
# Linear regression is often shown as:
# $$
# \overrightarrow{y}_i = \beta_0 + \beta_1 \times \overrightarrow{x}_1 + ... + \beta_k \times \overrightarrow{x}_k + \epsilon
# $$
#
# This representation describes the belief that the numeric value of each target is a linear combination of the $k$ features (predictors) of $X$, where the $\beta_0$ term adjusts the "baseline" value of the prediction (this means it is the prediction made when all other features equal $\overrightarrow{0}$).
#
#
#
# ## Prediction Representation
# The goal of supervised learning is to build a function that takes input batches of observations with the structure of $X_\text{batch}$ and produce vectors of predictions $\overrightarrow{p}_\text{batch} = \begin{pmatrix} p_1 \\ p_2 \\ \vdots \\ p_b \end{pmatrix}$ with values $p_i$. Each of these prediction numbers $p_i$ should be close to the target values $y_i$.
#
# For simplicity assume first there is no intercept term $\beta_0$. We can represent the output of a linear regression model as the *dot product* of each **observation vector** $\overrightarrow{x}_i = \begin{pmatrix} x_{i1} & x_{i2} & ... & x_{ik} \end{pmatrix}$ with another **vector of parameters** that we will call $W$:
# $$
# W = \begin{pmatrix}
#   w_1 \\
#   w_2 \\
#   \vdots \\
#   w_k
# \end{pmatrix}
# $$
#
# Then our prediction is written as:
# $$
# p_i = \overrightarrow{x}_i \times W = w_1 \cdot x_{i1} + w_2 \cdot x_{i2} + ... + w_k \cdot x_{ik}
# $$
#
# But we want to make predictions using linear regression with a *batch of observations* so we must use matrix multiplication instead of dot product to generate predictions. Given a batch of size $b$, the data matrix is:
# $$
# X_\text{batch} = \begin{pmatrix}
#   x_{11} & x_{12} & ... & x_{1k} \\
#   x_{21} & x_{22} & ... & x_{2k} \\
#   \vdots & \vdots & ... & \vdots \\
#   x_{b1} & x_{b2} & ... & x_{bk}
# \end{pmatrix}
# $$
# where ...
# * $k$ = number of data features or predictors
# * $n$ = number of rows or number of data observations
# * $b$ = number of batches
#
#
#
# ... then performing the *matrix multiplication* of this batch $X_\text{batch}$ with $W$ gives a *vector of predictions for the batch* as desired:
# $$
# \begin{aligned}
# p_\text{batch}
# :&= X_\text{batch} \times W  \\
# &= \begin{pmatrix}
#   x_{11} & x_{12} & ... & x_{1k} \\
#   x_{21} & x_{22} & ... & x_{2k} \\
#   \vdots & \vdots & ... & \vdots \\
#   x_{b1} & x_{b2} & ... & x_{bk}
# \end{pmatrix}
# \times
# \begin{pmatrix}
#   w_1 \\
#   w_2 \\
#   \vdots \\
#   w_k
# \end{pmatrix} \\
# &= \begin{pmatrix}
#   x_{11} \cdot w_1 & x_{12} \cdot w_2  & ... & x_{1k} \cdot w_k  \\
#   x_{21} \cdot w_1  & x_{22} \cdot w_2  & ... & x_{2k} \cdot w_k  \\
#   \vdots & \vdots & ... & \vdots \\
#   x_{b1} \cdot w_1  & x_{b2} \cdot w_2  & ... & x_{bk} \cdot w_k
# \end{pmatrix} \\
# &=: \begin{pmatrix}  p_1 \\ p_2 \\ \vdots \\ p_b \end{pmatrix}
# \end{aligned}
# $$
#
#
# To include an actual intercept term (which we will call the "bias"), we must add the intercept to the $\overrightarrow{p}_\text{batch}$. Each element of the model's prediction $p_i$ will be the dot product of the row observation vector $\overrightarrow{x}_i$ with each element in the parameter tensor $W$, added to the intercept $\overrightarrow{\beta_0}$:
# $\color{red}{\text{TODO: intercept vector issue}}$
# $$
# \begin{aligned}
# p_{\text{batch_with_bias}}
# :&= X_\text{batch} \times W + \overrightarrow{\beta_0} \\
# &= \begin{pmatrix}
#   x_{11} & x_{12} & ... & x_{1k} \\
#   x_{21} & x_{22} & ... & x_{2k} \\
#   \vdots & \vdots & ... & \vdots \\
#   x_{b1} & x_{b2} & ... & x_{bk}
# \end{pmatrix}
# \times
# \begin{pmatrix}
#   w_1 \\
#   w_2 \\
#   \vdots \\
#   w_k
# \end{pmatrix}
# +
# \begin{pmatrix}
#   \beta_0 \\ \beta_0 \\ \vdots \\ \beta_0
# \end{pmatrix} \\
# &= \begin{pmatrix}
#   x_{11} \cdot w_1 & x_{12} \cdot w_2  & ... & x_{1k} \cdot w_k  + \beta_0 \\
#   x_{21} \cdot w_1  & x_{22} \cdot w_2  & ... & x_{2k} \cdot w_k  + \beta_0 \\
#   \vdots & \vdots & ... & \vdots \\
#   x_{b1} \cdot w_1  & x_{b2} \cdot w_2  & ... & x_{bk} \cdot w_k  + \beta_0
# \end{pmatrix} \\
# &=: \begin{pmatrix}  p_1 \\ p_2 \\ \vdots \\ p_b \end{pmatrix}
# \end{aligned}
# $$
#
#
#
# ## Training the Model
# To compare the predictions to a standard, we need to use a vector of *targets* $\overrightarrow{y}_\text{batch}$ associated with the batch of observations $X_\text{batch}$ fed into the function. Then we compute a single number that is a function of $\overrightarrow{y}_\text{batch}$ and $\overrightarrow{p}_\text{batch}$ that represents the model's penalty for making predictions that it did. We can choose the **mean squared error** which is the average squared value that our model's predictions were off target:
# $$
# \begin{aligned}
# L &= \text{MSE}(\overrightarrow{p}_\text{batch}, \overrightarrow{y}_\text{batch})  \\
# &= \text{MSE} \Bigg( \begin{pmatrix} p_1 \\ p_2 \\ \vdots \\ p_b \end{pmatrix}, \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_b \end{pmatrix} \Bigg) \\
# &= \frac {(y_1 - p_1)^2 + (y_2 - p_2)^2 + ... + (y_b - p_b)^2 } {b}
# \end{aligned}
# $$
#
# This number $L$ is the loss and we will compute the *gradient* of this number with respect to each element of the parameter tensor $W$. In training the model, we will use these derivatives to update each element of $W$ in the direction that would cause $L$ to decrease.
#
# The loss value $L$ that we ultimately compute is:
# $$
# L = \Lambda(\nu(X, W), Y)
# $$
#
#
# ## Linear Regression: The Code
# Computing derivatives for nested functions using the chain rule involves two sets of steps: **first** we perform a "forward pass", passing the input successively forward through a series of operations and saving the quantities computed as we go; **second**, we use those quantities to compute the appropriate derivatives during the "backward pass".
#
# Below in the code, the forward function computes quantities in the forward pass and saves them in a dictionary, which serves the additional purpose of differentiating between forward pass quantities computed her and the parameters themselves.

# %%
def checkBroadcastable(x: Tensor, y: Tensor) -> bool:
    prependOnes = Tensor([1 for i in range(0, abs(x.ndim - y.ndim))])
    (smallestTensor, largestTensor) = (y, x) if y.ndim < x.ndim else (x, y)
    onesSmallestSize = torch.cat((prependOnes, Tensor(smallestTensor.size())), 0)
    pairs = list(zip(Tensor(largestTensor.size()).tolist(), onesSmallestSize.tolist() )) 
    batchDimPairs = pairs[0:-2] # all the dims except the last two are the batch dimension pairs
    isBroadcastable = all(map(lambda p: p[0] == 1 or p[1] == 1 or p[0] == p[1], batchDimPairs))

    return isBroadcastable

# %%
x = torch.randn(8,2,6,7,2,1,4,3, names = ('batch_one', 'batch_two', 'batch_three', 'batch_four', 'batch_five', 'batch_six', 'A', 'B'))
y = torch.randn(        1,5,3,2, names = ('batch_five', 'batch_six', 'C', 'D'))

assert checkBroadcastable(x, y)


x = torch.empty(5, 2, 4, 1)
y = torch.empty(   3, 1, 1)

assert not checkBroadcastable(x, y)


# %% codecell
def forwardLinearRegression(X_batch: Tensor, y_batch: Tensor, weights: Dict[str, Tensor]) -> Tuple[float, Dict[str, Tensor]]:
    '''
    Forward pass for the step-by-step linear regression.

    weights = dictionary of parameters, with parameters 1 through k under the key 'W' and intercept under key 'B'
    X_batch = data matrix of observations for a particular batch size.
        size == (b x k)
    W = weights or parameters
        size == (k x 1)
    y_batch = targets
        size == (b x 1)
    '''
    beta_0: float = weights['B']
    W: FloatTensor = weights['W'].type(FloatTensor)
    X_batch: FloatTensor = X_batch.type(FloatTensor)
    y_batch: FloatTensor = y_batch.type(FloatTensor)


    # Check batch sizes of X and y are equal:
    # TODO check if the batch size is ever not on the first dimension
    isBatchSizeConsistent = X_batch.size('batchSize') == y_batch.size('batchSize')
    assert isBatchSizeConsistent


    # Check: to multiply higher-dim tensors, then the tensors need to be broadcastable (means: all the dimensions before the last two should obey the broadcasting rules, see name inference tutorial)
    # isFirstPartEqualShape: bool = X_batch.shape[0 : X_batch.ndim - 2] == W.shape[0:W.ndim - 2]
    assert checkBroadcastable(X_batch, W)


    # Check: to do matrix multiplication, the last dim of X must equal the second-last dim of W.
    canDoMatMul: bool = X_batch.shape[-1] == W.shape[-2]
    assert canDoMatMul


    # TODO should I keep beta as 1x1 array?
    isInterceptANumber = beta_0.shape[0] == beta_0.shape[1] == 1
    assert isInterceptANumber


    ### Compute the forward pass operations
    N: Tensor = torch.matmul(X_batch, W)

    P: Tensor = N + beta_0 

    loss: Tensor = torch.mean(torch.pow(y_batch - P, 2))

    # Save the information computed on the forward pass
    forwardInfo: Dict[str, Tensor] = {}
    forwardInfo['X'] = X_batch 
    forwardInfo['N'] = N 
    forwardInfo['P'] = P 
    forwardInfo['y'] = y_batch 

    return loss, forwardInfo 
# %% codecell
