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

from sympy import det, Determinant, Trace, Transpose, Inverse, HadamardProduct, Matrix, MatrixExpr, Expr, Symbol, derive_by_array, Lambda, Function, MatrixSymbol, Identity,  Derivative, symbols, diff 

from sympy import tensorcontraction, tensorproduct
from sympy.physics.quantum.tensorproduct import TensorProduct

from sympy.abc import x, i, j, a, b, c

from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matmul import MatMul


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

#from IPython.display import HTML
#display(HTML('<style>.text_cell .CodeMirror{font-family:Qarmic sans}</style>'))
# %% markdown
# ## Linear Regression Formulas
#
# ### Data Representation
#
# The data is represented in a matrix $X$ with $n$ rows, each of which represents an observation with $k$ features, all of which are numbers. Each row observation is a vector $\overrightarrow{x}_i = \begin{pmatrix}x_{i1}  &  x_{i2}  &  x_{i3} & ... & x_{ik} \end{pmatrix}$. Each of these row observation vectors will be stacked on top of one another to form a $b \times k$ batch matrix, $X_{\text{batch}}$.
# %% [markdown]
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
# For simplicity assume first there is no intercept term $\beta_0$. We can represent the output of a linear regression model as the *dot product* of each **observation vector** $\overrightarrow{x}_i = \begin{pmatrix} x_{i1} & x_{i2} & ... & x_{ik} \end{pmatrix}$ with another **vector of parameters** that we will call $\overrightarrow{w}$:
# $$
# \overrightarrow{w} = \begin{pmatrix}
#   w_1 \\
#   w_2 \\
#   \vdots \\
#   w_k
# \end{pmatrix}
# $$
#
# Then our prediction is written as:
# $$
# p_i = \overrightarrow{x}_i \times \overrightarrow{w} = w_1 \cdot x_{i1} + w_2 \cdot x_{i2} + ... + w_k \cdot x_{ik}
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
# :&= X_\text{batch} \times \overrightarrow{w}  \\
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
# To include an actual intercept term (which we will call the "bias"), we must add the intercept to the $\overrightarrow{p}_\text{batch}$. Each element of the model's prediction $p_i$ will be the dot product of the row observation vector $\overrightarrow{x}_i$ with each element in the parameter vector $\overrightarrow{w}$, added to the intercept $\overrightarrow{\beta_0}$:
# $\color{red}{\text{TODO: intercept vector issue}}$
# $$
# \begin{aligned}
# p_{\text{batch_with_bias}}
# :&= X_\text{batch} \times \overrightarrow{w} + \overrightarrow{\beta_0} \\
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
#
# To compare the predictions to a standard, we need to use a vector of *targets* $\overrightarrow{y}_\text{batch}$ associated with the batch of observations $X_\text{batch}$ fed into the function. Then we compute a single number that is a function of $\overrightarrow{y}_\text{batch}$ and $\overrightarrow{p}_\text{batch}$ that represents the model's penalty for making predictions that it did. We can choose the **mean squared error** which is the average squared value that our model's predictions were off target:
# $$
# \begin{aligned}
# L &= \text{MSE}(\overrightarrow{p}_\text{batch_with_bias}, \overrightarrow{y}_\text{batch})  \\
# &= \text{MSE} \Bigg( \begin{pmatrix} p_1 \\ p_2 \\ \vdots \\ p_b \end{pmatrix}, \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_b \end{pmatrix} \Bigg) \\
# &= \frac {(y_1 - p_1)^2 + (y_2 - p_2)^2 + ... + (y_b - p_b)^2 } {b}
# \end{aligned}
# $$
#
# This number $L$ is the loss and we will compute the *gradient* of this number with respect to each element of the parameter tensor $\overrightarrow{w}$. In training the model, we will use these derivatives to update each element of $\overrightarrow{w}$ in the direction that would cause $L$ to decrease.





# %% [markdown]
# ## Linear Regression: The Code
#
# Computing derivatives for nested functions using the chain rule involves two sets of steps: **first** we perform a "forward pass", passing the input successively forward through a series of operations and saving the quantities computed as we go; **second**, we use those quantities to compute the appropriate derivatives during the "backward pass".
#
#
# ## Forward Pass for Linear Regression
#
# Below in the code, the forward function computes quantities in the forward pass and saves them in a dictionary, which serves the additional purpose of differentiating between forward pass quantities computed her and the parameters themselves.
#
# The equations that are computed in the forward pass are as follows:
#
# Let:
# * $k$ = number of data features or predictors
# * $n$ = number of rows or number of data observations
# * $b$ = number of batches
#
#
# ### Step 1: Matrix Multiplication: $N = \nu(X, \overrightarrow{w}) = X \times \overrightarrow{w}$
#
#
# Assume,
# * $X_\text{batch} \in \mathbb{R}^{b \times k}$
# * $\overrightarrow{w} \in \mathbb{R}^{k \times 1}$
# * $\nu : \mathbb{R}^{(b \times k) \times (k \times 1)} \rightarrow \mathbb{R}^{b \times 1}$
# * $N = \overrightarrow{N} \in \mathbb{R}^{b \times 1}$
#
#
# Multiply the matrix $X_\text{batch}$ and the vector of parameters $\overrightarrow{w}$
# $$
# \begin{aligned}
# \overrightarrow{N} &= \nu(X, \overrightarrow{w}) \\
# &= X_\text{batch} \times \overrightarrow{w} \\
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
# \end{pmatrix}
# \end{aligned}
# $$

# %% [markdown]
# ### Step 2: Predictions: $P = \alpha(N, B) = N + B$
#
#
# Assume:
# * $N = \overrightarrow{N} \in \mathbb{R}^{b \times 1}$
# * $B = \overrightarrow{\beta_0} \in \mathbb{R}$ so this intercept is a constant.
# $\color{red}{\text{TODO: vector intercept issue to fix}}$
# * $\alpha: \mathbb{R}^{(b \times 1) \times (1 \times 1)} \rightarrow \mathbb{R}^{b \times 1}$
# * $\overrightarrow{P} \in \mathbb{R}^{b \times 1}$
#
#
# $$
# \begin{aligned}
# \overrightarrow{P} &= \alpha(N, B) \\
# &= N + B \\
# &= \overrightarrow{N} + \overrightarrow{\beta_0} \\
# &= X_\text{batch} \times \overrightarrow{w} + \overrightarrow{\beta_0} \\
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

# %% [markdown]
# ### Step 3: Loss: $L = \Lambda(P, Y) = \text{MSE}(P, Y)$
#
# Assume:
# * $X \in \mathbb{R}^{b \times k}$
# * $\overrightarrow{w} \in \mathbb{R}^{k \times 1}$
# * $B = \overrightarrow{\beta_0} \in \mathbb{R}$ so this intercept is a constant.
# $\color{red}{\text{TODO vector intercept issue}}$
# * $Y = \overrightarrow{Y} \in \mathbb{R}^{b \times 1}$
# * $N = \overrightarrow{N} \in \mathbb{R}^{b \times 1}$
# * $\alpha: \mathbb{R}^{(b \times 1) \times (1 \times 1)} \rightarrow \mathbb{R}^{b \times 1}$
# * $P = \overrightarrow{P} \in \mathbb{R}^{b \times 1}$
# * $\Lambda: \mathbb{R}^{(b \times 1) \times (b \times 1)} \rightarrow \mathbb{R}$
# * $L \in \mathbb{R}$
#
#
#
# $$
# \begin{aligned}
# L &= \Lambda(\alpha(\nu(X, \overrightarrow{w}), B), Y) \\
#   &= \Lambda(\alpha(N, B), Y) \\
#   &= \Lambda(P, Y) \\
#   &= \text{MSE}(\overrightarrow{P}, \overrightarrow{Y})  \\
#   &= \text{MSE} \Bigg( \begin{pmatrix} p_1 \\ p_2 \\ \vdots \\ p_b \end{pmatrix}, \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_b \end{pmatrix} \Bigg) \\
#   &= \frac{(\overrightarrow{P} - \overrightarrow{Y})^2} {b} \\
#   &= \frac {\begin{pmatrix}
# \begin{pmatrix}  y_1 \\ y_2 \\ \vdots \\ y_b \end{pmatrix}
# -
# \begin{pmatrix}  p_1 \\ p_2 \\ \vdots \\ p_b \end{pmatrix}
#   \end{pmatrix}^2 } {b} \\
#   &= \frac {(y_1 - p_1)^2 + (y_2 - p_2)^2 + ... + (y_b - p_b)^2 } {b}
# \end{aligned}
# $$
#
#
# * **NOTE:** in the above steps, we use $N := \overrightarrow{N}$, $P := \overrightarrow{P} := \overrightarrow{p}_\text{batch_with_bias}$ and $Y := \overrightarrow{Y} := \overrightarrow{y}_\text{batch}$ and $X := X_\text{batch}$.
#   * $\color{red}{\text{Reason (?) maybe because we want to say that more batches can fit inside each of the symbols X, P, Y ...}}$

# %% codecell
def forwardLinearRegression(X_batch: Tensor, y_batch: Tensor, weights: Dict[str, Tensor]) -> Tuple[float, Dict[str, Tensor]]:
    '''
    Forward pass for the step-by-step linear regression.

    weights = dictionary of parameters, with parameters 1 through k under the key 'W' and intercept under key 'B'
    X_batch = data matrix of observations for a particular batch size.
        size == (b x k)
    w = weights or parameters
        size == (k x 1)
    y_batch = targets
        size == (b x 1)
    '''
    beta_0: float = weights['B']
    w: FloatTensor = weights['w'].type(FloatTensor)
    X_batch: FloatTensor = X_batch.type(FloatTensor)
    y_batch: FloatTensor = y_batch.type(FloatTensor)


    # Check batch sizes of X and y are equal:
    # TODO check if the batch size is ever not on the first dimension
    isBatchSizeConsistent = X_batch.size('batchSize') == y_batch.size('batchSize')
    assert isBatchSizeConsistent


    # Check: to multiply higher-dim tensors, then the tensors need to be broadcastable (means: all the dimensions before the last two should obey the broadcasting rules, see name inference tutorial)
    # isFirstPartEqualShape: bool = X_batch.shape[0 : X_batch.ndim - 2] == W.shape[0:W.ndim - 2]
    assert checkBroadcastable(X_batch, w)


    # Check: to do matrix multiplication, the last dim of X must equal the second-last dim of W.
    canDoMatMul: bool = X_batch.shape[-1] == w.shape[-2]
    assert canDoMatMul


    # TODO should I keep beta as 1x1 array?
    isInterceptANumber = beta_0.shape[0] == beta_0.shape[1] == 1
    assert isInterceptANumber


    ### Compute the forward pass operations
    N_batch: Tensor = X_batch.matmul(w)
    # torch.mv() only works if w.shape == (dim1, ) instaed of (dim1, dim2) so we have to use matmul here, even though w is a vector so dim2 == 1.

    # TODO: In pytorch it IS possible to add a tensor and scalar together but in a vector space it is NOT possible ...
    P_batch_with_bias: Tensor = N_batch + beta_0

    # This is the MSE formula:
    loss: Tensor = torch.mean(torch.pow(y_batch - P_batch_with_bias, 2))

    # Save the information computed on the forward pass
    forwardInfo: Dict[str, Tensor] = {}
    forwardInfo['X'] = X_batch
    forwardInfo['N'] = N_batch
    forwardInfo['P'] = P_batch_with_bias
    forwardInfo['y'] = y_batch

    return loss, forwardInfo






# %% [markdown]
# ## Training the Model: The Code
#
# We must compute $\frac{\partial L} {\partial w_i}$ for every parameter $w_i$ in the vector of parameters $\overrightarrow{w}$, and we must additionally compute $\frac{\partial L}{\partial \beta_0}$.
# $\color{red}{\text{TODO: intercept vector issue}}$
# The forward pass simply calculated the loss via a series of nested functions, and the backward pass will now compute the partial derivatives of each function, evaluating those derivatives at the functions' inputs and multiplying them together.
#
#
# ## Backward Pass for Linear Regression
# In the backward pass, we compute the derivative of each constituent function and evaluate those derivatives at the inputs that those functions received on the forward pass, and then multiply all these derivatives together as the chain rule dictates.
#
# The derivative product we must compute is:
# $$
# \begin{aligned}
# \frac{\partial L}{\partial \overrightarrow{w}}
#   &= \frac{\partial }{\partial P}\Lambda(P, Y) \times \frac{\partial }{\partial N} \alpha(N, B)  \times \frac{\partial }{\partial \overrightarrow{w}} \nu(X, \overrightarrow{w}) \\
#   &= \frac{\partial }{\partial \overrightarrow{P}} \Lambda(\overrightarrow{P}, \overrightarrow{Y}) \times \frac{\partial}{\partial \overrightarrow{N}} \alpha(\overrightarrow{N}, \overrightarrow{\beta_0}) \times \frac{\partial }{\partial \overrightarrow{w}} \nu(X, \overrightarrow{w}) \\
#   &= \frac{\partial \Lambda}{\partial \overrightarrow{P}} \times \frac{\partial \alpha}{\partial \overrightarrow{N}} \times \frac{\partial \nu}{\partial \overrightarrow{w}}
# \end{aligned}
# $$
#
#
# ### Calculating Loss Gradient $\frac{\partial \Lambda}{\partial \overrightarrow{w}}$:
#
# **The Direct Way**
#
# $$
# \frac{\partial \Lambda}{\partial \overrightarrow{w}} = \begin{pmatrix}
#   w_1 \\ w_2 \\ \vdots \\ w_k
# \end{pmatrix}
# $$
#
#
# **The Chain Rule Way**
#
# 1. **Step 1:** Compute $\frac{\partial \Lambda}{\partial \overrightarrow{P}}$
#
# $$
# \begin{aligned}
# \frac{\partial }{\partial \overrightarrow{P}} \Lambda(\overrightarrow{P}, \overrightarrow{Y})
#   &= \frac{\partial}{\partial \overrightarrow{P}} \Bigg( \frac{(\overrightarrow{Y} - \overrightarrow{P})^2} {b} \Bigg) \\
#   &= \frac{-1 \cdot (2 \cdot (\overrightarrow{Y} - \overrightarrow{P})) } {b}
# \end{aligned}
# $$


# %% [markdown]
# Preliminary variable Set-up:
# %%
# NOTE: using numbers to construct actual matrices
b_, k_, l_ = 5, 8, 1
# NOTE: using the symbolic shapes so that matrix derivative of MatrixSymbols Xm * wm can occur correctly
b, k, l = symbols('b k l', commutative=True)

# Variables
Xelem = Matrix(b_, k_, lambda i,j : var_ij('x', i, j))
X = MatrixSymbol('X', b, k)
Xs = Symbol('X', commutative=True)

welem = Matrix(k_, l_, lambda i,_ : var_i('w', i))
w = MatrixSymbol('\overrightarrow{w}', k, l)
ws = Symbol('\overrightarrow{w}', commutative=True)

# matrix variable for sympy Lambda function arguments
M = MatrixSymbol('M', i, j) #, is_commutative=True)# abstract shape


# N = X * w
v = Function("nu",applyfunc=True)
v_ = lambda a,b: a*b
#vL = Lambda((a,b), a*b)
VL = Lambda((X,w), MatrixSymbol('V', b, l) ) #Xm.shape[0], wm.shape[1]))
vN = lambda mat1, mat2: Matrix(mat1.shape[0], mat2.shape[1], lambda i,j: var_ij('n', i, j))

N = MatrixSymbol('\overrightarrow{N}', b, l) #, is_commutative=True)
Ns = Symbol('\overrightarrow{N}', commutative=True)
Nelem = vN(Xelem, welem)
Nspec = v_(Xelem, welem)
Nfunc = v(X,w)


# Alpha = N + B = X*w + beta_0
B = MatrixSymbol('\overrightarrow{\\beta_0}', b , l)
Bs = Symbol('beta_0', is_commutative=True)
Belem = Matrix(b_, l_, lambda i, j : Bs)
# TODO when N is symbol use right arrow overtop? For B too or not? Be consistent.

sigma = Function('sigma', commutative=True)
sigma_ = lambda matrix : matrix.applyfunc(sigma)


alpha = Function('alpha', applyfunc=True, commutative=True)
alpha_ = lambda N_, B_: N_ + B_ # take in matrix symbols to add them.
# Function to avoid errors with replacing in alpha two-arg function - just subtitute the desired matrix result right in alpha (problem is cannot do this straight from alpha or Pfunc because alpha has two arguments and for some reason error is thrown ...)
alphaSub = lambda X, w, B: alpha(sigma(v(X, w)).replace(v, v_).replace(sigma, sigma_).diff(w), B)
#alphaL = Lambda( (N, B), N + B)
#alphaApply = Lambda( (N, B), alpha(N, B))
#alphaL = Lambda( ())

Pfunc = alpha(Nfunc, B)
P = MatrixSymbol('\overrightarrow{P}', b, l)
Ps = Symbol('\overrightarrow{P}', commutative=True)
Pelem = Matrix(b_, l_, lambda i, _: var_i('p', i))
Pspec = Pfunc.subs({X: Xelem, w:welem}).replace(v, v_).replace(B, Belem).replace(alpha, alpha_)
#alpha_(Nspec, B)


# Lambda function: L = Lambda(P, Y)
Y = MatrixSymbol('\overrightarrow{Y}', b, l)
Ys = Symbol('\overrightarrow{Y}', commutative=True)
Yelem = Matrix(b_, l_, lambda i, _: var_i('y', i))

lambd = Function("lambda", commutative=True)
lambd_ = lambda P_, Y_ : (Y_ - P_)**2 / b #Takes matrix symbols to calculate MSE

#L = lambd(alpha(v(Xs, ws), B), Ys)
#Lfunc = lambda X_mat, w_vec, B, Y_vec: lambd(alpha(v(X_mat, w_vec), B), Y_vec)
#L = Lfunc(Xm, wm, B, Ym)
L = lambd(alpha(v(X, w), B), Y)
# TODO: need this symbol, non-matrix version because otherwise replacing and derivating results in error" noncommutative scalars in matmul are not supported (even if ANY one of the arguments here are Matrix or MatrixSymbol)
# TODO so now I don't know whether the following derivations with the SYMBOLS instead of MATRIX SYMBOL forms (as in ch1) are actually correct (meaning I don't think the correct matrix calculus rules are being applied)
Ls = lambd(alpha(v(Xs, ws), Bs), Ys)

showGroup([L, Ls])


# %% codecell
dL_dw_overallAbstract = Ls.diff(ws)

showGroup([
    dL_dw_overallAbstract,
    dL_dw_overallAbstract.replace(v(Xs, ws), Ns),
    dL_dw_overallAbstract.replace(alpha(v(Xs, ws), Bs), Ps),
    dL_dw_overallAbstract.replace(alpha(v(Xs, ws), Bs), Ps).replace(v(Xs, ws), Ns)
])

# %%
# dL_dw_abstract = # TODO how to get the matrices rules to apply like in Ch1?
# NOTE: interesting here sympy knew how to apply the product rule (WHY, is it just because there is only one kind of argument, the X? Does it not work when two kinds like A and B?)


# %%
sigma(v(X, X.T)).replace(v, v_).replace(sigma, sigma_).diff(X)

# %%
A = MatrixSymbol('A', k, a)
showGroup([
    sigma(v(X, A)).replace(v, v_).replace(sigma, sigma_).diff(X),
    sigma(v(X, A)).replace(v, v_).replace(sigma, sigma_).diff(A)
])

# %%
showGroup([
    sigma(v(X, w)).replace(v, v_).replace(sigma, sigma_).diff(X),
    sigma(v(X, w)).replace(v, v_).replace(sigma, sigma_).diff(w)
])
# %%
showGroup([
    Ls.replace(v, v_).diff(ws),
    Ls.replace(v, v_).diff(ws).replace(alpha(Xs*ws, Bs), Ps)
])

# %%
#alpha(sigma(v(X, w)), Bs).replace(v, v_).replace(sigma, sigma_).diff(w)
# %%
#alpha(*[sigma(v(X,w)), Bs])
alphaSub(X, w, B)
# NTOTE: below is not what I want, need the X and w to be both  matrix symbols so the transpose happens (correct matrix calc)
# alpha(*[sigma(v(X,w)), Bs]).replace(v, v_).replace(sigma, sigma_).replace(w, ws).diff(ws)
# %%
_d = Symbol('d', commutative=True)
Lambda(_d, sigma(_d).diff(_d))







# %% codecell
# Chain rule component way:
dL_dP = Ls.diff(alpha(v(Xs, ws), Bs))
# Ls.replace(alpha(v(Xs, ws), Bs), Ps).diff(Ps)

showGroup([
    dL_dP,
    dL_dP.replace( alpha(v(Xs, ws), Bs), Ps),
    dL_dP.replace( alpha(v(Xs, ws), Bs), Ps).replace(lambd, lambd_),
    dL_dP.replace( alpha(v(Xs, ws), Bs), Ps).replace(lambd, lambd_).doit()
])
# %% codecell
dP_dN = alpha(v(Xs, ws), Bs).diff(v(Xs, ws))

showGroup([
    dP_dN,
    dP_dN.replace(v(Xs, ws), Ns),
    dP_dN.replace(v(Xs, ws), Ns).replace(alpha(Ns, Bs), Ps),
    dP_dN.replace(v(Xs, ws), Ns).replace(alpha, alpha_).replace(Bs, B).replace(Ns, N),
    dP_dN.replace(v(Xs, ws), Ns).replace(alpha, alpha_).replace(Bs, B).replace(Ns, N).doit()
])
# %%
# TODO left off here need to find a way to make this a matrix of ones not a tensor of tensors.

ex = alpha(Nelem, Belem).replace(alpha, alpha_).diff(Nelem)
#ex.shape
contr = tensorcontraction(ex, (0,))
#contr.shape
contr2 = tensorcontraction(contr, (0,))
#contr2.shape


### TODO URGENT HERE
# TODO try the old way matrix deriv then do the tensor contraction to see if the result is the same with multiplying by parts manually and substituting the matrix there.
showGroup([
    ex, contr, contr2
])
# %% codecell
dN_dw = v(Xs, ws).subs({Xs:X, ws:w}).replace(v, v_).diff(w)

dN_dw # TODO fix not correct, need to actually evaluate this. 
# %%
# Putting them together:
dL_dP * dP_dN * dN_dw
# %%
