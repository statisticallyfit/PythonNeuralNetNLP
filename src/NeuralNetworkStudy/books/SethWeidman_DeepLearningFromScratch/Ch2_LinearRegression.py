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
# ## Prediction Representation
# The goal of supervised learning is to build a function that takes input batches of observations with the structure of $X_\text{batch}$ and produce vectors of predictions $\overrightarrow{p}_\text{batch} = \begin{pmatrix} p_1 \\ p_2 \\ \vdots \\ p_b \end{pmatrix}$ with values $p_i$. Each of these prediction numbers $p_i$ should be close to the target values $y_i$. 
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
# %% codecell
