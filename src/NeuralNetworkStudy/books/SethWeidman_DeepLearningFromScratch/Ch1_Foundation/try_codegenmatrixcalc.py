

# %% codecell

# CODEGEN TUTORIAL SOURCE: https://docs.sympy.org/latest/modules/codegen.html?highlight=tensorproduct#sympy.codegen.array_utils.CodegenArrayTensorProduct



# %%
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
from src.MatrixCalculusStudy.MatrixDerivLib.diff import matrixDifferential
#from src.MatrixCalculusStudy.MatrixDerivLib.printingLatex import myLatexPrinter

# For displaying
from IPython.display import display, Math
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax') #, latex_printer= lambda e, **kw: myLatexPrinter.doprint(e))

# %%
A = MatrixSymbol("A", a, c)
J = MatrixSymbol("J", c, a)
B = MatrixSymbol("B", c, b)
R = MatrixSymbol("R", c,c)
C = MatrixSymbol('C', b, b)
D = MatrixSymbol('D', b, a)
L = MatrixSymbol('L', a, c)
E = MatrixSymbol('E', c, b)
G = MatrixSymbol('G', b, b) # pair of C
H = MatrixSymbol('H', b, c)
K = MatrixSymbol('K', a, b) # pair of D

b, k, l = symbols('b k l', commutative=True)
X = MatrixSymbol('X', b, k)
w = MatrixSymbol('\overrightarrow{w}', k, l)

# Testing with real numbers because the matrix diff function needs real number dimensions 
# TODO make diffmatrix convert symbolic dims into real dims that match just for the sake of keeping symbolic dims at the end (then replace)
A_ = MatrixSymbol("A", 4, 3)
J_ = MatrixSymbol("J", 3, 4)
B_ = MatrixSymbol("B", 3, 2)
R_ = MatrixSymbol("R", 3,3)
C_ = MatrixSymbol('C', 2, 2)
D_ = MatrixSymbol('D', 2, 4)
L_ = MatrixSymbol('L', 4, 3)
E_ = MatrixSymbol('E', 3, 2)
G_ = MatrixSymbol('G', 2, 2) # pair of C
H_ = MatrixSymbol('H', 2, 3)
K_ = MatrixSymbol('K', 4, 2) # pair of D

X_ = MatrixSymbol('X', 2, 5)
w_ = MatrixSymbol('\overrightarrow{w}', 5, 6)









# %%
from sympy.codegen.array_utils import CodegenArrayPermuteDims, CodegenArrayElementwiseAdd, CodegenArrayContraction, CodegenArrayTensorProduct, recognize_matrix_expression, nest_permutation, parse_indexed_expression

from sympy.combinatorics import Permutation

from sympy import Sum



# %% codecell
# PARSING: Trace 

expr = Sum(R[i,i], (i, 0, c - 1))

cgExpr = parse_indexed_expression(expr)

showGroup([
    expr, 
    cgExpr, 
    recognize_matrix_expression(cgExpr)
])
# %%
# PARSING: More complex traces: 
expr = Sum(A[i, j] * J[j, i], (i, 0, a-1), (j, 0, c-1))

cgexpr = parse_indexed_expression(expr)

rec = recognize_matrix_expression(cgexpr)

showGroup([
    expr, 
    cgexpr, 
    rec
])


# %% codecell
# PARSING: Extraction of diagonal --- DID NOT WORK
expr = R[i, i]

cgexpr = parse_indexed_expression(expr)

showGroup([
    expr, 
    cgexpr, 
    recognize_matrix_expression(cgexpr)
])




# %% codecell
# ADDITION
recognize_matrix_expression(CodegenArrayElementwiseAdd(A + L))




# %% codecell
# PARSING: Matrix multiplication
expr = Sum(A[i, j] * B[j, k], (j, 0, c - 1))

cgexpr = parse_indexed_expression(expr)

rec = recognize_matrix_expression(cgexpr)

showGroup([
    expr, 
    cgexpr, 
    rec
])

assert cgexpr.shape == (A.shape[0], B.shape[1])
assert all(map(lambda a: isinstance(a, MatrixSymbol), rec.args))
assert rec.shape == (A.shape[0], B.shape[1])

# %% codecell
# Specify that k = starting index (result is transpose of product while before it is just matrix product)
expr = Sum(A[i, j] * B[j, k], (j, 0, c - 1))

cgexpr = parse_indexed_expression(expr, first_indices=[k])

rec = recognize_matrix_expression(cgexpr)

showGroup([
    expr, cgexpr, rec
])

# %%
# PARSING: Symbolic matrix multiplication (not from indexing notation)
expr = A*R*B

cgexpr = CodegenArrayContraction.from_MatMul(expr)

rec = recognize_matrix_expression(cgexpr)

showGroup([
    expr, 
    cgexpr, 
    rec
])

# %%
# PARSING: Matrix Multiplication (resulting in multiple matrix factors, as specified by the passed dimension tuples)
cgexpr = CodegenArrayContraction(CodegenArrayTensorProduct(A, B, C, D), (1,2), (5,6))

rec = recognize_matrix_expression(cgexpr)

showGroup([
    cgexpr, rec
])
# NOTE: meaning of the indices
# Each matrix is assigned an index: 
# 0_A_1
# 2_B_3
# 4_C_5
# 6_D_7
# So specifying which matrices get multiplied implies passing in the tuples holding the dims that correspond to each matrix side. Ex: (3,4) means to multiply: B*C




# %%
# PARSING: Tensor product
expr = TensorProduct(A, B)

cg = CodegenArrayContraction.from_MatMul(expr)

rec = recognize_matrix_expression(cg)

showGroup([
    expr, 
    cg, 
    rec
    # TODO why does this result in simple matrix multiplication? Tensor product is NOT regular matrix multiplication!
])




# %%
# TRANSPOSE
recognize_matrix_expression(CodegenArrayPermuteDims(A, [1, 0]))
# %%
# PARSING: Nested Transpose
cg = CodegenArrayPermuteDims(CodegenArrayTensorProduct(A, R), [1, 0, 3, 2])

nested = nest_permutation(cg)

showGroup([
    cg, 
    recognize_matrix_expression(cg),
    nested,
    recognize_matrix_expression(nested)
])
# %%
# TODO try to get dims for tensor contraction 



# %% 
# PARSING: Transpose (multiple ways)
expr = Sum(C[j, i] * G[j, k], (j, 0, b - 1))

cg = parse_indexed_expression(expr)

rec = recognize_matrix_expression(cg)

showGroup([expr, cg, rec])
# %%
# PARSING: Transpose (multiple ways)
expr = Sum(C[i, j] * G[k, j], (j, 0, b - 1))

cg = parse_indexed_expression(expr)

rec = recognize_matrix_expression(cg)

showGroup([expr, cg, rec])
# %%
# PARSING: Transpose (multiple ways)
expr = Sum(C[j, i] * G[k, j], (j, 0, b - 1))

cg = parse_indexed_expression(expr)

rec = recognize_matrix_expression(cg)

showGroup([expr, cg, rec])
# %%
# %%
# PARSING: Transpose (multiple ways)
expr = Sum(C[j,i] * G[k, j], (j, 0, b - 1))

cg = parse_indexed_expression(expr, first_indices= [k])

rec = recognize_matrix_expression(cg)

showGroup([expr, 
    cg, 
    rec, 
    rec.doit()
])



# %% codecell
# PARSING: Mix of multiplication and Trace

# TRANSPOSED DIMS:  (a, c) * (c,c) * (b, c)
# REGULAR DIMS:     (a, c) * (c, c) * (c, b)
#expr = Sum(A[i, j] * R[k, j] * H[l, k], (j, 0, c-1), (k, 0, c-1))

# TRANSPOSED DIMS:  (a, c) * (b, c) * (a, b) 
# REGULAR DIMS:     (a,c) * (c, b) * (b, a)

expr = Sum(A[i, j] * H[k, j] * K[l, k], (j, 0, c-1), (k, 0, b-1))

# NOTE: need to put the interval endings so that the iterators inside the array correspond to the transposed dims locations

cgexpr = parse_indexed_expression(expr)

rec = recognize_matrix_expression(cgexpr)

showGroup([
    expr, 
    cgexpr, 
    rec
])


