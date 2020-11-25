
import numpy as np
from numpy import ndarray

from typing import *
import itertools
from functools import reduce


from sympy import det, Determinant, Trace, Transpose, Inverse, HadamardProduct, TensorProduct, Matrix, MatrixExpr, Expr, Symbol, derive_by_array, Lambda, Function, MatrixSymbol, Identity,  Derivative, symbols, diff, tensorcontraction

from sympy.abc import x, i, j, a, b, c

from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matmul import MatMul

from sympy.physics.quantum.tensorproduct import TensorProduct

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
def derivMatadd(expr: MatrixExpr, byVar: MatrixSymbol) -> MatrixExpr: 
    assert isinstance(expr, MatAdd), "The expression is not of type MatAdd"

    # Split at the plus / minus sign
    components: List[MatMul] = list(expr.args)

    assert all(map(lambda comp : isinstance(comp, MatMul), components)), "All componenets are not MatMul"

    # Filter all components and make sure they have the byVar argument inside. If they don't keep them out to signify that derivative is 0. (Ex: d(A)/d(C) = 0)
    componentsToDeriv: List[MatMul] = list(filter(lambda c: c.has(A), components))

# TODO left off here matrix add so that expression like AB + D^T can be differentiated (by A) to equal B^T and another one AB + A should result in B^T + I (assuming appropriate sizes). 



# %%
# TODO 1) incorporate rules from here: https://hyp.is/6EQ8FC5YEeujn1dcCVtPZw/en.wikipedia.org/wiki/Matrix_calculus
# TODO 2) incorporate rules from Helmut book
# TODO 3) test-check with matrixcalculus.org and get its code ...?
def derivMatmul(expr: MatrixExpr, 
    #mat1: MatrixSymbol, mat2: MatrixSymbol, 
    byVar: MatrixSymbol) -> MatrixExpr: 
    
    #assert (isinstance(expr, MatMul) or isinstance(expr, MatrixExpr)) and not isinstance(expr, MatAdd), "The expression is of type MatAdd - this is the wrong function to which to pass this expression."

    # STEP 1: if the arguments have symbolic shape, then need to create fake ones for this function (since the R Matrix will replace a MatrixSymbol M, and we need actual numbers from the arguments' shapes to construct the Matrix R, which represents their multiplication)

    # Get the dimensions
    syms: List[MatrixSymbol] = list(expr.free_symbols)
    # Get shape tuples
    dimTuples: List[Tuple[Symbol, Symbol]] = list(map(lambda s: s.shape, syms))
    # Remove the tuples, just get all dimensions in a flat list. 
    dimsFlat: List[Symbol] = [dimension for shapeTuple in dimTuples for dimension in shapeTuple]
    # Get alll the dimensions multiplied into one product
    dimsProduct: Expr = foldLeft(operator.mul, 1, dimsFlat)
    #mat1.shape + mat2.shape + byVar.shape)
    # Get all the unique symbols from the product of dimension symbols
    dims: Set[Symbol] = dimsProduct.free_symbols

    isAnyDimASymbol = any(map(lambda dim_i : isinstance(dim_i, Symbol), dims))

    #any(map(lambda dim_i: isinstance(dim_i, Symbol), expr.shape))

    if isAnyDimASymbol: 
        #(DIM_1, DIM_2, DIM_3) = (5, 2, 3) # 

        # # Choose some arbitrary integer dimensions: (to be used later for substitution)
        ds: List[int] = list(range(2, len(dims) + 2, 1))
        numDims: List[int] = list(np.random.choice(ds, len(dims), replace=False))
        # Map the integer dimensions to each symbol dimension (dict)
        symdimToNumdim: Dict[Symbol, int] = dict(zip(dims, numDims))
        # Create its inverse for later use
        numdimToSymdim: Dict[int, Symbol] = dict( [(v, k) for k, v in symdimToNumdim.items()] )

        # Now make the matrix symbols using those fake integer dimensions: 
        syms_ = [MatrixSymbol(m.name, symdimToNumdim[m.shape[0]], symdimToNumdim[m.shape[1]] ) for m in syms]

        # Create a dictionary mapping the symbol-dim matrices to the num-dim matrices
        symmatToNummat: Dict[MatrixSymbol, MatrixSymbol] = dict(zip(syms, syms_))
        # Make inverse of the symbolic-dim to num-dim dictionary: 
        nummatToSymmat: Dict[MatrixSymbol, MatrixSymbol] = dict( [(v, k) for k, v in symmatToNummat.items()] )

        # Create another expression from the given argument expr, but with all symbols having the number dimension not symbol dimension (as was given)

        expr_: MatrixExpr = expr.xreplace(symmatToNummat)

        byVar_ : MatrixSymbol = MatrixSymbol(byVar.name, symdimToNumdim[byVar.shape[0]], symdimToNumdim[byVar.shape[1]])

    else: 
        # If any of the dims are NOT symbols then it means all of the dims are numbers, so just rename as follows: 
        (expr_, byVar_) = (expr, byVar)

    # STEP 2: create the intermediary replacer functions
    t = Function('t', commutative=True)
    # Create the lambd apply func
    M = MatrixSymbol('M', i, j) # abstract shape, doesn't matter
    tL = Lambda(M, M.applyfunc(t)) # apply the miniature inner function to the matrix (to get that lambda symbol using an arbitrary function t())

    # Create shape of the resulting matrix multiplication of arguments. Will use this to substitute; shape must match because it interacts with derivative involving the arguments, which have related shape. 
    R = Matrix(MatrixSymbol('R', *expr_.shape) )
    # TODO when using this example in link below, error occurs here because Trace has no attribute shape. TODO fix 
    # TEST EXAMPLE: https://hyp.is/JQLtqi23EeuTI_v0yX2T9Q/www.kannon.link/free/category/research/mathematics/
    # TODO seems that diff(expr) can be done well when the expr is a Trace? Or a sympy function?

    # STEP 3: Do derivative: 
    deriv = t(expr_).replace(t, tL).diff(byVar_)

    #cutMatrix = diff(tL(M), M).subs(M, R).doit()

    #derivWithCutExpr = deriv.xreplace({expr_ : R}).doit()


    # TODO don't know if this is correct as matrix calculus rule. 
    # # Create the invisible matrix to substitute in place of the lambda expression thing (substituting just 1 works when expr_ = A * B, simple matrix product, and substituting results in correct answer: d(A*B)/dB = A^T * 1 = A^T but not sure if the result is correct here for arbitrary matrix expression expr)
    IDENT_ = MatrixSymbol('I', *expr_.shape)
    INVIS_ = MatrixSymbol(' ', *expr_.shape)
    # Create the symbolic-dim companion for later use
    IDENT = MatrixSymbol('I', *expr.shape) #numdimToSymdim[INVIS_.shape[0]], numdimToSymdim[INVIS_.shape[1]])
    INVIS = MatrixSymbol(' ', *expr.shape)

    # TODO (this is the questionable part) Substitute the expression to cut with the invisible, correctly-shaped matrix
    #derivResult_ = derivWithCutExpr.xreplace({cutMatrix : INVIS_}).doit()


    # Another way to find derivative by replacement (not using matrix but instead replacing the lambda directly:)
    cutLambda_ = diff(tL(M), M).subs(M, expr_)
    derivResult_ = deriv.xreplace({cutLambda_ : IDENT_}).doit()

    # STEP 4: replace the original variables if any dimension was a symbol, so that the result expression dimensions are still symbols
    if isAnyDimASymbol:
        # Add the invisible matrix: 
        nummatToSymmat_with_invis = dict( list(nummatToSymmat.items() ) + [(IDENT_, IDENT)] )

        derivResult: MatrixExpr = derivResult_.xreplace(nummatToSymmat_with_invis)
        # NOTE: use xreplace when want to replace all the variables all at ONCE (this seems to be the effect with xreplace rather than subs). Else replacing MatrixSymbols happens one-by-one and alignment error gets thrown. 
        #derivResult_.subs(nummatToSymmat_with_invis)

        # Asserting that all dims now are symbols, as we want: 
        assert all(map(lambda dim_i: isinstance(dim_i, Symbol), derivResult.shape))

        # TODO questionable hacky action here (ref: seth book pg 43 and 54): If the expr was a product of two matrices then result should not have the "I" matrix but should equal transpose of other matrix: 
        exprIsProdOfTwoMatrices = len(expr.args) == 2 and isinstance(expr, MatMul) and all(map(lambda anArg: isinstance(anArg, MatrixSymbol), expr.args))

        #if exprIsProdOfTwoMatrices:
        #    derivResult: MatrixExpr = derivResult.xreplace({IDENT : INVIS})

        return derivResult

    exprIsProdOfTwoMatrices = len(expr.args) == 2 and isinstance(expr, MatMul) and all(map(lambda anArg: isinstance(anArg, MatrixSymbol), expr.args))

    #if exprIsProdOfTwoMatrices:
    #    derivResult_: MatrixExpr = derivResult_.xreplace({IDENT_ : INVIS_}) 

    return derivResult_ # else if no dim was symbol, return the num-dim result. 
# %%
A = MatrixSymbol("A", a, c)
J = MatrixSymbol("J", c, a)
B = MatrixSymbol("B", c, b)
R = MatrixSymbol("R", c,c)
C = MatrixSymbol('C', b, b)
D = MatrixSymbol('D', b, a)
L = MatrixSymbol('L', a, c)
E = MatrixSymbol('E', c, b)

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

X_ = MatrixSymbol('X', 2, 5)
w_ = MatrixSymbol('\overrightarrow{w}', 5, 6)
# %%
derivMatmul(A * B, A)
# %%
derivMatmul(A * B, B)
# %% 
derivMatmul(X*w, w)
# %% 
derivMatmul(X*w, X)
# %%
derivMatmul(Inverse(R), R)

# TODO result seems exactly the same as this calculator gives except my result here does not say it is tensor product, just wrongly assumes matrix product. 
# TODO figure out which is correct
# %%
derivMatmul(A, A)
# TODO this doesn't work
# %%
derivMatmul(A + A, A)
# %%
derivMatmul(A*B - D.T, A)
# %%
derivMatmul(B * Inverse(C) * E.T * L.T * A * E * D, E)
# %%
derivMatmul(B * Inverse(C) * E.T * L.T * A * E * D, C)
# %%
derivMatmul(B * Inverse(C) * E.T * L.T * A * E * D, A)
# %%
derivMatmul(B * Inverse(C) * E.T * L.T * A * E * D, D)
# %%
derivMatmul(B * Inverse(C) * E.T * L.T * A * E * D, L)
# %%
diffMatrix(B_ * Inverse(C_) * E_.T * L_.T * A_ * E_ * D_, C_)
# %%
derivMatmul(B_ * Inverse(C_) * E_.T * L_.T * A_ * E_ * D_,   E_)

# %% 
derivMatmul(X*Y*A*X*B, A)


# %%
diffMatrix(X, X)
# %%
diffMatrix(a*X, X)
# %%
# TODO error with cyclic permute 
# # diffMatrix(Trace(R_), R_)

diff(Trace(R), R)
# %%
#diff(Trace(X*w), X)
diff(Trace(A*J), J)
# %%
diff(Trace(A*J), A)
# %%
diffMatrix(A*J, A)
# %%
derivMatmul(A*J, A)





# %%
# TODO fix this function so it can take symbolic matrices
#diffMatrix(B * Inverse(C) * E.T * L.T * A * E * D,   E)

diffMatrix(B_ * Inverse(C_) * E_.T * L_.T * A_ * E_ * D_,   E_)

# TODO this function seems very wrong: just seems to add the differential operator to the byVar instead of actually doing anything to the expression. 

# TODO: this function's result doesn't match the derivMatmul result of this expression
# %%
f = Function('f', commutative=True)
g = Function('g', commutative=True)

#derivMatexpr(f(A) * g(A), A)
#diffMatrix(f(A) * g(A), A)
# TODO I removed the application function part so this won't work anymore (need to add it back in)

# %% codecell
