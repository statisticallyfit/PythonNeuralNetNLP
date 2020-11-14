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


# %% 