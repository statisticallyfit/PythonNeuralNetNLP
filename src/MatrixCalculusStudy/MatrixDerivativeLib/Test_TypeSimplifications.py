
# %%
import numpy as np
from numpy import ndarray

from typing import *
import itertools
from functools import reduce


from sympy import det, Determinant, Trace, Transpose, Inverse, Function, Lambda, HadamardProduct, Matrix, MatrixExpr, Expr, Symbol, derive_by_array, MatrixSymbol, Identity,  Derivative, symbols, diff

from sympy import srepr , simplify

from sympy import tensorcontraction, tensorproduct, preorder_traversal
from sympy.functions.elementary.piecewise import Undefined
from sympy.physics.quantum.tensorproduct import TensorProduct

from sympy.abc import x, i, j, a, b, c

from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matmul import MatMul

from sympy.core.numbers import NegativeOne, Number

from sympy.core.assumptions import ManagedProperties

# %%
import torch
import torch.tensor as tensor

# Types
MatrixType = ManagedProperties

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



from src.utils.GeneralUtil import *

from src.MatrixCalculusStudy.MatrixDifferentialLib.symbols import Deriv
from src.MatrixCalculusStudy.MatrixDifferentialLib.diff import matrixDifferential
from src.MatrixCalculusStudy.MatrixDifferentialLib.printingLatex import myLatexPrinter


# For displaying
from IPython.display import display, Math
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax', latex_printer= lambda e, **kw: myLatexPrinter.doprint(e))


from src.MatrixCalculusStudy.MatrixDerivativeLib.TypeSimplifications import * 



# %%
def mataddDeriv(expr: MatrixExpr, byVar: MatrixSymbol) -> MatrixExpr:
    assert isinstance(expr, MatAdd), "The expression is not of type MatAdd"

    # Split at the plus / minus sign
    components: List[MatMul] = list(expr.args)

    assert all(map(lambda comp : isinstance(comp, MatMul), components)), "All componenets are not MatMul"

    # Filter all components and make sure they have the byVar argument inside. If they don't keep them out to signify that derivative is 0. (Ex: d(A)/d(C) = 0)
    componentsToDeriv: List[MatMul] = list(filter(lambda c: c.has(A), components))

# TODO NOT DONE YET with this function, left off here matrix add so that expression like AB + D^T can be differentiated (by A) to equal B^T and another one AB + A should result in B^T + I (assuming appropriate sizes).



# %%
# TODO 1) incorporate rules from here: https://hyp.is/6EQ8FC5YEeujn1dcCVtPZw/en.wikipedia.org/wiki/Matrix_calculus
# TODO 2) incorporate rules from Helmut book
# TODO 3) test-check with matrixcalculus.org and get its code ...?
def matmulDeriv(expr: MatrixExpr,
    #mat1: MatrixSymbol, mat2: MatrixSymbol,
    byVar: MatrixSymbol) -> MatrixExpr:

    #assert (isinstance(expr, MatMul) or isinstance(expr, MatrixExpr)) and not isinstance(expr, MatAdd), "The expression is of type MatAdd - this is the wrong function to which to pass this expression."

    # STEP 1: if the arguments have symbolic shape, then need to create fake ones for this function (since the R Matrix will replace expr MatrixSymbol M, and we need actual numbers from the arguments' shapes to construct the Matrix R, which represents their multiplication)

    # Get the dimensions
    syms: List[MatrixSymbol] = list(expr.free_symbols)
    # Get shape tuples
    dimTuples: List[Tuple[Symbol, Symbol]] = list(map(lambda s: s.shape, syms))
    # Remove the tuples, just get all dimensions in expr flat list.
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

        # Create expr dictionary mapping the symbol-dim matrices to the num-dim matrices
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
    # TODO seems that diff(expr) can be done well when the expr is expr Trace? Or expr sympy function?

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

    # STEP 4: replace the original variables if any dimension was expr symbol, so that the result expression dimensions are still symbols
    if isAnyDimASymbol:
        # Add the invisible matrix:
        nummatToSymmat_with_invis = dict( list(nummatToSymmat.items() ) + [(IDENT_, IDENT)] )

        derivResult: MatrixExpr = derivResult_.xreplace(nummatToSymmat_with_invis)
        # NOTE: use xreplace when want to replace all the variables all at ONCE (this seems to be the effect with xreplace rather than subs). Else replacing MatrixSymbols happens one-by-one and alignment error gets thrown.
        #derivResult_.subs(nummatToSymmat_with_invis)

        # Asserting that all dims now are symbols, as we want:
        assert all(map(lambda dim_i: isinstance(dim_i, Symbol), derivResult.shape))

        # TODO questionable hacky action here (ref: seth book pg 43 and 54): If the expr was expr product of two matrices then result should not have the "I" matrix but should equal transpose of other matrix:
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
matmulDeriv(A * B, A)
# %%
matmulDeriv(A * B, B)
# %%
matmulDeriv(X*w, w)
# %%
matmulDeriv(X*w, X)
# %%
matmulDeriv(Inverse(R), R)

# TODO result seems exactly the same as this calculator gives except my result here does not say it is tensor product, just wrongly assumes matrix product.
# TODO figure out which is correct
# %%
matmulDeriv(A, A)
# TODO this doesn't work
# %%
matmulDeriv(A + A, A)
# %%
matmulDeriv(A*B - D.T, A)
# %%
matmulDeriv(B * Inverse(C) * E.T * L.T * A * E * D, E)
# %%
matmulDeriv(B * Inverse(C) * E.T * L.T * A * E * D, C)
# %%
matmulDeriv(B * Inverse(C) * E.T * L.T * A * E * D, A)
# %%
matmulDeriv(B * Inverse(C) * E.T * L.T * A * E * D, D)
# %%
matmulDeriv(B * Inverse(C) * E.T * L.T * A * E * D, L)
# %%
#diffMatrix(B_ * Inverse(C_) * E_.T * L_.T * A_ * E_ * D_, C_)
# %%
matmulDeriv(B_ * Inverse(C_) * E_.T * L_.T * A_ * E_ * D_,   E_)

# %%




diff(Trace(R), R)
# %%
#diff(Trace(X*w), X)
diff(Trace(A*J), J)
# %%
diff(Trace(A*J), A)
# %%
matmulDeriv(A*J, A)



# %%
f = Function('f', commutative=True)
f_ = lambda mat: sum(mat)

g = Function('g', commutative=True)
g_a = Function('g_a', commutative=True)
g_ = lambda mat: mat.applyfunc(g)


v = Function('nu', commutative=True)
v_ = lambda a, b: a * b
#derivMatexpr(f(A) * g(A), A)
#diffMatrix(f(A) * g(A), A)
# TODO I removed the application function part so this won't work anymore (need to add it back in)

# %%
# NOTE Need to replace with the non-symbolic dim matrices X_ and w_ instead of the symbolic dim matrices X and w because else we can't apply Matrix to the X*w product thus we can't apply lambda f function as below:
compose(f, g_a, v)(X_, w_).replace(v, v_).replace(g_a, g_).replace(f, f_)

# %% codecell
compose(f, g_a, v)(Matrix(X_), Matrix(w_)).replace(v, v_).replace(g_a, g_)
# %% codecell
# Display error (xi subscript thing)
#compose(f, g_a, v)(X_, w_).replace(v, v_).replace(g_a, g_).replace(f, f_).diff(Matrix(w_)).doit()

# %% codecell
compose(f, g_a, v)(X_, w_).replace(v, v_).replace(g_a, g_).diff(w_)











# %% codecell

# TODO debugging the functions in 'simplifications.py' file, normally wouldn't need to import here so erase after done
from src.MatrixCalculusStudy.MatrixDerivLib.simplifications import _conditional_replace, cyclic_permute_dX_cond, cyclic_permute_dX_repl, _cyclic_permute


# %%
# STRATEGY 3: DIFFERENTIAL APPROACH -------------------------------

from src.MatrixCalculusStudy.MatrixDerivLib.symbols import d

#de = diffMatrix(A*R*J, R)
#de
# %%
# TODO left off here to fix these functions

showGroup([
    _conditional_replace(Trace(A * d(R) * J), cyclic_permute_dX_cond(d(R)), cyclic_permute_dX_repl(d(R))),

    _conditional_replace(Inverse(A * d(R) * J), cyclic_permute_dX_cond(d(R)), cyclic_permute_dX_repl(d(R))),

    _conditional_replace(Transpose(A * d(R) * J), cyclic_permute_dX_cond(d(R)), cyclic_permute_dX_repl(d(R))),
])


# %%

# Examples of how permutating happens in different situations, just for testing:

B_sq = MatrixSymbol('B', a, a)
C_sq = MatrixSymbol('C', a, a)
E_sq = MatrixSymbol('E', a, a)
L_sq = MatrixSymbol('L', a, a)
A_sq = MatrixSymbol('A', a, a)
D_sq = MatrixSymbol('D', a, a)


exprDA = B_sq * Inverse(C_sq) * E_sq.T * L_sq.T * d(A_sq) * E_sq * D_sq

exprDB = d(B_sq) * Inverse(C_sq) * E_sq.T * L_sq.T * d(A_sq) * E_sq * D_sq

exprInvDC = B_sq * d(Inverse(C_sq)) * E_sq.T * L_sq.T * d(A_sq) * E_sq * D_sq

showGroup([
    exprDA,

    _conditional_replace(exprDA, cyclic_permute_dX_cond(d(A_sq)), cyclic_permute_dX_repl(d(A_sq))),

    exprDB,

    _conditional_replace(exprDB, cyclic_permute_dX_cond(d(B_sq)), cyclic_permute_dX_repl(d(B_sq))),

    exprInvDC,

    _conditional_replace(exprInvDC, cyclic_permute_dX_cond(d(Inverse(C_sq))), cyclic_permute_dX_repl(d(Inverse(C_sq))))

])









# %% ----------------------------------------------------------
# TEST DATA FOR BELOW FUNCTIONS:

a, b, c = symbols('expr b c', commutative = True)

C = MatrixSymbol('C', c, c)
D = MatrixSymbol('D', c, c)
E = MatrixSymbol('E', c, c)
A = MatrixSymbol('A', c, c)
B = MatrixSymbol('B', c, c)
R = MatrixSymbol('R', c, c)
J = MatrixSymbol('J', c, c)
M = MatrixSymbol('M', c, c)
K = MatrixSymbol("K", c, c)

# %%

# SPLIT ONCE TESTS: 

L = MatrixSymbol('L', c, c)

expr = C*E*B*E*L*E*D

assert splitOnce(expr.args, E, 1) == ([C], [B, E, L, E, D])
assert splitOnce(expr.args, E, 2) == ([C, E, B], [L, E, D])
assert splitOnce(expr.args, E, 3) == ([C, E, B, E, L], [D])
assert splitOnce(expr.args, E, 0) == ([], [])
assert splitOnce(expr.args, E, 4) == ([], [])
# TODO how to assert error for negative number n?


# %% ------------------------------------------------------------


# CHUNK TYPES TEST 1: 


# Constructors
cs = [Inverse, Transpose, Transpose, Inverse, Inverse, Inverse, Transpose, MatrixSymbol, Symbol, Symbol, Symbol, NegativeOne, NegativeOne, NegativeOne, NegativeOne, Inverse, Symbol, Transpose, Inverse, Symbol, Transpose, Inverse, Inverse, MatMul, MatMul, MatAdd]

res = chunkTypesBy(byTypes = [Transpose, Inverse], types = cs)
check = [
    [Inverse, Transpose, Transpose, Inverse, Inverse, Inverse, Transpose],
    [MatrixSymbol, Symbol, Symbol, Symbol, NegativeOne, NegativeOne, NegativeOne, NegativeOne],
    [Inverse],
    [Symbol],
    [Transpose, Inverse],
    [Symbol],
    [Transpose, Inverse, Inverse],
    [MatMul, MatMul, MatAdd]
]

assert res == check


# %% ------------------------------------------------------------

# STACK TEST 1: 

ts = [Inverse, Inverse, Transpose, Inverse, Transpose, Transpose, Inverse, MatMul, MatAdd]

cts = [Transpose, Transpose, Transpose, Inverse, Inverse, Inverse, Inverse, MatMul, MatAdd]

assert stackTypesBy(byType = Transpose, types = ts) == cts 



# %% -------------------------------------------------------------


# INNER TEST 1: simplest case possible, just matrixsymbol as innermost nesting
t1 = Transpose(Inverse(Inverse(Inverse(Transpose(Transpose(A))))))
assert inner(t1) == A


# INNER TEST 2: second nesting inside the innermost expression
t2 = Transpose(Inverse(Transpose(Inverse(Transpose(Inverse(Inverse(MatMul(B, A, Transpose(Inverse(A*C)) ) )))))))
c2 = MatMul(B, A, Transpose(Inverse(A*C)))

assert inner(t2) == c2


# INNER TEST 3: testing the real purpose now: the inner() function must get just the first level of innermosts:
# NOTE: expr is from test 14 expr
t3 = Transpose(Inverse(
    MatMul(A*D.T*R.T ,
        Transpose(
            Inverse(
                MatMul(C.T, E.I, L, B.T)
            )
        )
    )
))
c3 = t3.arg.arg #just twice depth
assert inner(t3) == c3




# %% -------------------------------------------


# DIGGER TEST 1:
expr = Transpose(Inverse(Inverse(MatMul(
    Inverse(Transpose(Inverse(R*C*D))),
    Inverse(Transpose(Inverse(Transpose(B*A*R))))
))))
check = [
    ([Transpose, Inverse, Inverse], expr.arg.arg.args[0]),
    ([Inverse, Transpose, Inverse], R*C*D),
    ([Inverse, Transpose, Inverse, Transpose], B*A*R)
]

testCase(algo = digger, expr = expr, check = check, byType = None)



# %% -------------------------------------------------------------


# INNER TRAIL TEST 1
assert innerTrail(A) == ([], A)
# %%
# INNER TRAIL TEST 2
assert innerTrail(A.T) == ([Transpose], A)

assert innerTrail(Inverse(Transpose(Inverse(Inverse(A))))) == ([Inverse, Transpose, Inverse, Inverse], A)
# %%
# INNER TRAIL TEST 3
L = Transpose(Inverse(B*A*R))

expr = Transpose(Inverse(
    MatMul(A*D.T*R.T ,
        Transpose(
            Inverse(
                MatMul(C.T, E.I, L, B.T)
            )
        )
    )
))
# Showing how innerTrail gets only the first level while digger gets all the levels (in the order of preorder traversal, as it happens)
assert innerTrail(expr) == digger(expr)[0]

# %%
# INNER TRAIL TEST 4

# This is the result of grouptranspose then algotransposeripple of expression from 14.expr test
tg = Transpose(Transpose(Inverse(MatMul(
    Transpose(Inverse(MatMul(
        B, Inverse(B*A*R), Transpose(Inverse(E)), C
    ))),
    Transpose(Transpose(MatMul(R, D, A.T)))
))))

# Showing how inner gets just the innermost arg like innerTrail but without the types.
assert inner(tg) == innerTrail(tg)[1]

# Showing how inner trail gets just the first level
assert innerTrail(tg) == digger(tg)[0]


# %% --------------------------------------------------------------


# NOTE: "single" means individual matrixsymbol
# NOTE: "grouped" means symbols gathered with an operation (+, -, *) and WRAPPED IN  constructors (trace, transpose, inverse ...)
def testCase(algo, expr, check: MatrixExpr, byType: MatrixType = None):
    params = [byType, expr] if not (byType == None) else [expr]
    res = algo(*params)

    showGroup([
        expr, res, check 
    ])
    #assert expr.doit() == res.doit() 
    try:
        assert equal(expr, res)
    except Exception: 
        hasMatPow = lambda e: "MatPow" in srepr(res)

        print("ASSERTION ERROR: equal(expr, res) did not work")
        print("Had MatPow: ", hasMatPow(res))

    try:
        assert res.doit() == check.doit() 
    except Exception: 
        print("ASSERTION ERROR: res.doit() == check.doit() --- maybe MatPow ? ")

# %%
def testGroupCombineAdds(expr, check: MatrixExpr, byType: MatrixType):
    res = group(byType = byType, expr = expr, combineAdds = True)

    showGroup([
        expr, res, check 
    ])

    assert equal(expr, res)
    assert res.doit() == check.doit()
# %% -------------------------------------------------------------


# # TEST 1a: SL + GA = single symbol + grouped MatAdd    
expr_SLaGA = MatAdd(A, Transpose(B + C.T) )
# %%
check = expr_SLaGA

testCase(algo = group, expr = expr_SLaGA, check = check, byType = Transpose)
# %%

check = Transpose( (B + C.T) + A.T )

testGroupCombineAdds(expr = expr_SLaGA, check = check, byType = Transpose)
# %%
# TODO see the todo for matadd general addends error in the rippleout function
check = expr_SLaGA

testCase(rippleOut, expr = expr_SLaGA, check = expr_SLaGA, byType = Transpose)

# %%

check = expr_SLaGA

testCase(algo = factor, expr = expr_SLaGA, check = check, byType = Transpose)
# %%

# NOTE not liking the polarize result here - got too complicated. # Fixed with countTopTransp inner function inside polarize()

#check_TOFIX = Transpose(MatAdd(
#    Transpose(MatAdd(C, B.T)), A.T
#))
check = expr_SLaGA

testCase(algo = polarize, expr = expr_SLaGA, check = check, byType = Transpose)

# %% --------------------------------------------------------------


# TEST 1b: SL + GA = single symbol + grouped MatAdd (with more layerings per addend)

expr_SLaGA = MatAdd(
    Inverse(Transpose(Transpose(A))),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        B , Inverse(Transpose(Transpose(E))) , C.T , R
    ))))) )
)
# %%

check = MatAdd(
    Inverse(Transpose(Transpose(A))),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        Inverse(Transpose(Transpose(E))), B, R, C.T
    ))))))
)
testCase(algo = group, expr = expr_SLaGA, check = check, byType = Transpose)
# %%


check = MatAdd(
    Transpose(Transpose(Inverse(A))),
    Transpose(Transpose(Transpose(Inverse(Inverse(MatAdd(
        Transpose(Transpose(Inverse(E))), B, R, C.T
    ))))))
)

testCase(algo = rippleOut, expr = expr_SLaGA, check = check, byType = Transpose)
# %%


check = MatAdd(
    A.I, 
    Transpose(Inverse(Inverse(MatAdd(
        E.I, B, R, C.T
    ))))
)
testCase(algo = factor, expr = expr_SLaGA, check = check, byType = Transpose)

# %%


check = Transpose(MatAdd(
    Transpose(Inverse(A)), 
    Inverse(Inverse(E.I + B + R + C.T))
))

testCase(algo = polarize, expr = expr_SLaGA, check = check, byType = Transpose)
# %% --------------------------------------------------------------



# TEST 2a: SL * GA = single symbol * grouped MatAdd

expr_SLmGA = MatMul(
    A,
    Inverse(Transpose(MatAdd(B, C.T)))
)
# %%

# TODO result got too complicated
check = Transpose(MatMul(
    Transpose(Inverse(Transpose(MatAdd(B, C.T)))), 
    A.T
))

testCase(algo = group, expr = expr_SLmGA, check = check, byType = Transpose)
# %%


check = MatMul(
    A,
    Transpose(Inverse(MatAdd(B, C.T)))
)

testCase(algo = rippleOut, expr = expr_SLmGA, check = check, byType = Transpose)
# %%


check = MatMul(
    A,
    Transpose(Inverse(MatAdd(B, C.T)))
)
testCase(algo = factor, expr = expr_SLmGA, check = check, byType = Transpose)

# %%


#check_TOFIX = Transpose(MatMul(
#    Transpose(Inverse(MatAdd(C, B.T))), 
#    A.T
#))
check = MatMul(
    A, 
    Transpose(Inverse(MatAdd(B, C.T)))
)
testCase(algo = polarize, expr = expr_SLmGA, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 2b: SL * GA = single symbol * grouped MatAdd (with more layerings per addend)

expr_SLmGA = MatMul(
    Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) )
)
# %%


check = Transpose(MatMul(
    
    Transpose(Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) )),
    Transpose(Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))))
))

testCase(algo = group, expr = expr_SLmGA, check = check, byType = Transpose)
# %%


check = MatMul(
    Transpose(Transpose(Transpose(Transpose(Inverse(Inverse(Inverse(A))))))),
    Transpose(Transpose(Transpose(Inverse(Inverse(MatAdd(
        B , Transpose(Transpose(Inverse(E))) , C.T , Transpose(Inverse(R))
    ))))) )
)

testCase(algo = rippleOut, expr = expr_SLmGA, check = check, byType = Transpose)
# %%



check = MatMul(
    Inverse(Inverse(Inverse(A))),
    Transpose(Inverse(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    ))))
)
testCase(algo = factor, expr = expr_SLmGA, check = check, byType = Transpose)
# %%

# NOTE: fixed with countTopTransp inner function inside polarize()
# check_TOFIX 
check = MatMul(
    Inverse(Inverse(Inverse(A))), 
    Transpose(Inverse(Inverse(MatAdd(
        E.I, B, Transpose(Inverse(R)), C.T
    ))))
)
testCase(algo = polarize, expr = expr_SLmGA, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 3a: SL + GM = single symbol + grouped MatMul

expr_SLaGM = MatAdd(A, MatMul(B, A.T, R.I))
# %%


check  = MatAdd(
    A, 
    Transpose(MatMul(
        Transpose(Inverse(R)), A, B.T
    ))
)
testCase(algo = group, expr = expr_SLaGM, check = check, byType = Transpose)
# %%


check = expr_SLaGM

testCase(algo = factor, expr = expr_SLaGM, check = check, byType = Transpose)

testCase(algo = rippleOut, expr = expr_SLaGM, check = check, byType = Transpose)

testCase(algo = polarize, expr = expr_SLaGM, check = exprcheck_SLaGM, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 3b: SL + GM = single symbol + grouped MatMul (with more layerings per addend)

expr_SLaGM = MatAdd(
    Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) )
)
# %%

# STOPPING testing group() here because these tests are mainly meant for factor and polarize, group just complicates things, not that sophisticated!
# %%



check = MatAdd(
    Transpose(Transpose(Transpose(Transpose(Inverse(Inverse(Inverse(A))))))), 
    Transpose(Transpose(Transpose(Inverse(Inverse(MatAdd(
        B, Transpose(Inverse(R)), C.T, Transpose(Transpose(Inverse(E)))
    ))))))
)
testCase(algo = rippleOut, expr = expr_SLaGM, check = check, byType = Transpose)
# %%


check = MatAdd(
    Inverse(Inverse(Inverse(A))),
    Transpose(Inverse(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    ))))
)

testCase(algo = factor, expr = expr_SLaGM, check = check, byType = Transpose)

testCase(algo = polarize, expr = expr_SLaGM, check = check, byType = Transpose)

# %% --------------------------------------------------------------



# TEST 4a: SL * GM = single symbol * grouped MatMul

expr_SLmGM = MatMul(A, MatMul(B.T, A, R.I))

# %%


check = expr_SLmGM

testCase(algo = rippleOut, expr = expr_SLmGM, check = check, byType = Transpose)

testCase(algo = factor, expr = expr_SLmGM, check = check, byType = Transpose)

testCase(algo = polarize, expr = expr_SLmGM, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 4b: SL * GM = single symbol * grouped MatMul (with more layerings per multiplicand)

expr_SLmGM = MatMul(
    Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatMul(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) )
)
# %%


check = MatMul(
    Transpose(Transpose(Transpose(Transpose(Inverse(Inverse(Inverse(A))))))), 
    Transpose(Transpose(Transpose(Inverse(Inverse(MatMul(
        B, Transpose(Transpose(Inverse(E))), C.T, Transpose(Inverse(R))
    ))))))
)
testCase(algo = rippleOut, expr = expr_SLmGM, check = check, byType = Transpose)
# %%


check = MatMul(
    Inverse(Inverse(Inverse(A))),
    Transpose(Inverse(Inverse(MatMul(
        B, E.I, C.T, Transpose(Inverse(R))
    ))))
)

testCase(algo = factor, expr = expr_SLmGM, check = check, byType = Transpose)
# %%
# NOTE: comparing the initial factor (p, result of polarize) with the factored of wrap-deep (fe) and with polarize of fe --- same thing as the initial factoring (p), so no need in this case to call polarize() again, can just stick with initial factor

#p = polarize(Transpose, expr_SLmGM)
#w = wrapDeep(Transpose, p)
#fe = factor(Transpose, w)
#polarize(Transpose, fe)
check = MatMul(
    Inverse(Inverse(Inverse(A))),
    Transpose(Inverse(Inverse(MatMul(
        B, E.I, C.T, Transpose(Inverse(R))
    ))))
)

testCase(algo = polarize, expr = expr_SLmGM, check = check, byType = Transpose)
# %% --------------------------------------------------------------



# TEST 5a: SA + GA = single symbol Add + grouped Matadd

expr_SAaGA = MatAdd(
    A.T, B.I, C,
    Inverse(Transpose(Inverse(MatAdd(
        B, C.T, D.I
    ))))
)
# %%

check = MatAdd(
    A.T, C, B.I, 
    Transpose(Inverse(Inverse(MatAdd(B, C.T, D.I))))
)
testCase(algo = rippleOut, expr = expr_SAaGA, check = check, byType = Transpose)
# %%


check = MatAdd(
    A.T, B.I, C,
    Transpose(Inverse(Inverse(MatAdd(
        B, C.T, D.I
    ))))
)

testCase(algo = factor, expr = expr_SAaGA, check = check, byType = Transpose)

testCase(algo = polarize, expr = expr_SAaGA, check = check, byType = Transpose)
# %% --------------------------------------------------------------



# TEST 5b: SA + GA = single symbol Add + grouped Matadd (with more layerings per component)

expr_SAaGA = MatAdd(
    Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
    Inverse(Inverse(B)),
    Transpose(Transpose(C)),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) )
)
# %%


check = MatAdd(
    Inverse(Inverse(B)),
    Transpose(Transpose(Transpose(Inverse(Inverse(MatAdd(
        Transpose(Transpose(Inverse(E))), B, Transpose(Inverse(R)), C.T
    )))))), 
    Transpose(Transpose(Transpose(Transpose(Inverse(Inverse(Inverse(A))))))), 
    Transpose(Transpose(C))
)
testCase(algo = rippleOut, expr = expr_SAaGA, check = check, byType = Transpose)
# %%


check = MatAdd(
    Inverse(Inverse(Inverse(A))),
    Inverse(Inverse(B)),
    C,
    Transpose(Inverse(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    ))))
)

testCase(algo = factor, expr = expr_SAaGA, check = check, byType = Transpose)

# %%


check = MatAdd(
    Inverse(Inverse(Inverse(A))),
    Inverse(Inverse(B)),
    C,
    Inverse(Inverse(MatAdd(
        R.I, C, Transpose(Inverse(E)), B.T
    )))
)
testCase(algo = polarize, expr = expr_SAaGA, check = check, byType = Transpose)
# %% --------------------------------------------------------------



# TEST 6a: SA * GA = single symbol Add * grouped Matadd

expr_SAmGA = MatMul(
    A.T, B.I, C,
    Inverse(Transpose(Inverse(MatAdd(
        B, C.T, D.I
    ))))
)
# %%


check = MatMul(
    A.T, B.I, C,
    Transpose(Inverse(Inverse(MatAdd(
        B, C.T, D.I
    ))))
)
testCase(algo = rippleOut, expr = expr_SAmGA, check = check, byType = Transpose)

testCase(algo = factor, expr = expr_SAmGA, check = check, byType = Transpose)

testCase(algo = polarize, expr = expr_SAmGA, check = check, byType = Transpose)
# %% --------------------------------------------------------------



# TEST 6b: SA * GA = single symbol Add * grouped Matadd (with more layerings per component)

expr_SAmGA = MatMul(
    Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
    Inverse(Inverse(B)),
    Transpose(Transpose(C)),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) )
)
# %%


check = MatMul(
    Transpose(Transpose(Transpose(Transpose(Inverse(Inverse(Inverse(A))))))), 
    Inverse(Inverse(B)),
    Transpose(Transpose(C)), 
    Transpose(Transpose(Transpose(Inverse(Inverse(MatAdd(
        B, Transpose(Transpose(Inverse(E))), C.T, Transpose(Inverse(R))
    ))))))
)

testCase(algo = rippleOut, expr = expr_SAmGA, check = check, byType = Transpose)
# %%



check = MatMul(
    Inverse(Inverse(Inverse(A))),
    Inverse(Inverse(B)),
    C,
    Transpose(Inverse(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    ))))
)

testCase(algo = factor, expr = expr_SAmGA, check = check, byType = Transpose)
# %%


check = MatMul(
    Inverse(Inverse(Inverse(A))),
    Inverse(Inverse(B)),
    C,
    Inverse(Inverse(MatAdd(
        R.I, C, Transpose(Inverse(E)), B.T
    )))
)
testCase(algo = polarize, expr = expr_SAmGA, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 7a: SA + SM = single symbol Add + single symbol Mul

expr_SAaSM = MatAdd(
    A.T, B.I, C,
    Inverse(Transpose(Inverse(MatMul(
        B, C.T, D.I, E
    ))))
)
# %%

check = MatAdd(
    A.T, B.I, C,
    Transpose(Inverse(Inverse(MatMul(
        B, C.T, D.I, E
    ))))
)
testCase(algo = rippleOut, expr = expr_SAaSM, check = check, byType = Transpose)

testCase(algo = factor, expr = expr_SAaSM, check = check, byType = Transpose)

testCase(algo = polarize, expr = expr_SAaSM, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 7b: SA + SM = single symbol Add + single symbol Mul (with more layerings per component)

expr_SAaSM = MatAdd(
    Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
    Inverse(Inverse(B)),
    Transpose(Transpose(C)),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatMul(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) ),
    Inverse(Transpose(Inverse(Transpose(MatMul(
        B, Inverse(Transpose(Transpose(Transpose(A)))), R
    )))))
)
# %%


check = MatAdd(
    Inverse(Inverse(Inverse(A))),
    Inverse(Inverse(B)),
    C,
    Transpose(Inverse(Inverse(MatMul(
        B, E.I, C.T, Transpose(Inverse(R))
    )))),
    Inverse(Inverse(MatMul(
        B, Transpose(Inverse(A)), R
    )))
)

testCase(algo = factor, expr = expr_SAaSM, check = check, byType = Transpose)

# STOPPPING rippleout tests here, too useless since we have factor to also do that while factoring. 
# %%



check = MatAdd(
    Inverse(Inverse(Inverse(A))),
    Inverse(Inverse(B)),
    C,
    Inverse(Inverse(MatMul(
        R.I, C, Transpose(Inverse(E)), B.T
    ))),
    Inverse(Inverse(MatMul(
        B, Transpose(Inverse(A)), R
    )))
)

testCase(algo = polarize, expr = expr_SAaSM, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 8a: SM + GA = single symbol Mul + group symbol Add

expr_SMaGA = MatAdd(
    MatMul(A.T, B.I, C, D.T),
    Inverse(Transpose(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)
# %%


check = MatAdd(
    MatMul(A.T, B.I, C, D.T),
    Transpose(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

testCase(algo = factor, expr = expr_SMaGA, check = check, byType = Transpose)

testCase(algo = polarize, expr = expr_SMaGA, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 8b: SM + GA = single symbol Mul + group symbol Add (with more layerings per component)

expr_SMaGA = MatAdd(
    MatMul(
        Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
        Inverse(Inverse(B)),
        Transpose(Transpose(C)),
        D.I
    ),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) ),
    Inverse(Transpose(Inverse(Transpose(MatAdd(
        B, Inverse(Transpose(Transpose(Transpose(A)))), R
    )))))
)
# %%


check = MatAdd(
    MatMul(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ),
    Transpose(Inverse(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    )))),
    Inverse(Inverse(MatAdd(
        B, Transpose(Inverse(A)), R
    )))
)

testCase(algo = factor, expr = expr_SMaGA, check = check, byType = Transpose)
# %%


check = MatAdd(
    MatMul(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ),
    Inverse(Inverse(MatAdd(
        R.I, C, Transpose(Inverse(E)), B.T
    ))),
    Inverse(Inverse(MatAdd(
        B, Transpose(Inverse(A)), R
    )))
)

testCase(algo = polarize, expr = expr_SMaGA, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 9a: SM * GA = single symbol Mul * group symbol Add

expr_SMmGA = MatMul(
    MatMul(A.T, B.I, C, D.T),
    Inverse(Transpose(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)
# %%


check = MatMul(
    MatMul(A.T, B.I, C, D.T),
    Transpose(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

testCase(algo = factor, expr = expr_SMmGA, check = check, byType = Transpose)
# %%


check = MatMul(
    MatMul(A.T, B.I, C, D.T),
    Inverse(Inverse(MatAdd(
        R.I, C, Transpose(Inverse(E)), B.T
    )))
)

testCase(algo = polarize, expr = expr_SMmGA, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 9b: SM * GA = single symbol Mul * group symbol Add (with more layerings per component)

expr_SMmGA = MatMul(
    MatMul(
        Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
        Inverse(Inverse(B)),
        Transpose(Transpose(C)),
        D.I
    ),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) ),
    Inverse(Transpose(Inverse(Transpose(MatAdd(
        B, Inverse(Transpose(Transpose(Transpose(A)))), R
    )))))
)
# %%


check = MatMul(
    MatMul(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ),
    Transpose(Inverse(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    )))),
    Inverse(Inverse(MatAdd(
        B, Transpose(Inverse(A)), R
    )))
)

testCase(algo = factor, expr = expr_SMmGA, check = check, byType = Transpose)
# %%


check = MatMul(
    MatMul(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ),
    Inverse(Inverse(MatAdd(
        R.I, C, Transpose(Inverse(E)), B.T
    ))),
    Inverse(Inverse(MatAdd(
        B, Transpose(Inverse(A)), R
    )))
)

testCase(algo = polarize, expr = expr_SMmGA, check = check, byType = Transpose)

# %% --------------------------------------------------------------


# TEST 10a: SM + GM = single symbol Mul + group symbol Mul

expr_SMaGM = MatAdd(
    MatMul(A.T, B.I, C, D.T),
    Inverse(Transpose(MatMul(
        A.T, B, E.I, Inverse(Transpose(R)), C.T
    )))
)
# %%


check = MatAdd(
    MatMul(A.T, B.I, C, D.T),
    Transpose(Inverse(MatMul(
        A.T, B, E.I, Transpose(Inverse(R)), C.T
    )))
)

testCase(algo = factor, expr = expr_SMaGM, check = check, byType = Transpose)
# %%

# NOTE: required second application of polarize
check_GOOD = MatAdd(
    Inverse(MatMul(C, R.I, Transpose(Inverse(E)), B.T, A)),
    MatMul(A.T, B.I, C, D.T)
)

testCase(algo = polarize, expr = expr_SMaGM, check = check_GOOD, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 10b: SM + GM = single symbol Mul + group symbol Mul (with more layerings per component)

expr_SMaGM = MatAdd(
    MatMul(
        Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
        Inverse(Inverse(B)),
        Transpose(Transpose(C)),
        D.I
    ),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatMul(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) ),
    Inverse(Transpose(Inverse(Transpose(MatMul(
        B, Inverse(Transpose(Transpose(Transpose(A)))), R
    )))))
)
# %%



check = MatAdd(
    MatMul(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ),
    Transpose(Inverse(Inverse(MatMul(
        B, E.I, C.T, Transpose(Inverse(R))
    )))),
    Inverse(Inverse(MatMul(
        B, Transpose(Inverse(A)), R
    )))
)

testCase(algo = factor, expr = expr_SMaGM, check = check, byType = Transpose)
# %%



check = MatAdd(
    MatMul(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ),
    Inverse(Inverse(MatMul(
        R.I, C, Transpose(Inverse(E)), B.T
    ))),
    Inverse(Inverse(MatMul(
        B, Transpose(Inverse(A)), R
    )))
)

testCase(algo = polarize, expr = expr_SMaGM, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 11a: SM * GM = single symbol Mul * group symbol Mul

expr_SMmGM = MatMul(
    MatMul(A.T, B.I, C, D.T),
    Inverse(Transpose(MatMul(
        A.T, B, E.I, Inverse(Transpose(R)), C.T
    )))
)
# %%


check = MatMul(
    MatMul(A.T, B.I, C, D.T),
    Transpose(Inverse(MatMul(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

testCase(algo = factor, expr = expr_SMmGM, check = check, byType = Transpose)
# %%


check_GOOD = MatMul(
    MatMul(A.T, B.I, C, D.T),
    Inverse(MatMul(
        C, R.I, Transpose(Inverse(E)), B.T, A
    ))
)

testCase(algo = polarize, expr = expr_SMmGM, check = check_GOOD, byType = Transpose)


# %% --------------------------------------------------------------


# TEST 11b: SM * GM = single symbol Mul * group symbol Mul (with more layerings per component)

expr_SMmGM = MatMul(
    MatMul(
        Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
        Inverse(Inverse(B)),
        Transpose(Transpose(C)),
        D.I
    ),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatMul(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) ),
    Inverse(Transpose(Inverse(Transpose(MatMul(
        B, Inverse(Transpose(Transpose(Transpose(A)))), R
    )))))
)
# %%



check = MatMul(
    MatMul(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ),
    Transpose(Inverse(Inverse(MatMul(
        B, E.I, C.T, Transpose(Inverse(R))
    )))),
    Inverse(Inverse(MatMul(
        B, Transpose(Inverse(A)), R
    )))
)

testCase(algo = factor, expr = expr_SMmGM, check = check, byType = Transpose)
# %%


check = MatMul(
    MatMul(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ),
    Inverse(Inverse(MatMul(
        R.I, C, Transpose(Inverse(E)), B.T
    ))),
    Inverse(Inverse(MatMul(
        B, Transpose(Inverse(A)), R
    )))
)

testCase(algo = polarize, expr = expr_SMmGM, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 12a: GA + GM = group symbol Add + group symbol Mul

expr_GAaGM = MatAdd(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), C.T
    )))
)

check = MatAdd(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Transpose(Inverse(MatMul(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

testCase(algo = factor, expr = expr_GAaGM, check = check, byType = Transpose)

testCase(algo = polarize, expr = expr_GAaGM, check = check, byType = Transpose)
# %% -------------------------------------------------------------
# TODO left off here refactoring:  (STAR)
# TODO: check wrapdeep isSym because the above expressions be.ir.t.i*c.t haven't been getting wrapped properly ... see then if you still need to apply the polarize twice



# TEST 12 b: GA + GM = group symbol Add + group symbol Mul (but just small change from C.T to C.T.T)

expr_GAaGM = MatAdd(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), Transpose(Transpose(C))
    )))
)
check_polarize = 

# %% --------------------------------------------------------------


# TEST 12b: GA + GM = group symbol Add + group symbol Mul (with more layerings per component)

expr_GAaGM = MatAdd(
    Inverse(Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
        Inverse(Inverse(B)),
        Transpose(Transpose(C)),
        D.I
    ))))))),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatMul(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) ),
    Inverse(Transpose(Inverse(Transpose(MatMul(
        B, Inverse(Transpose(Transpose(Transpose(A)))), R
    )))))
)
# %%


check = MatAdd(
    Transpose(Inverse(Inverse(Inverse(MatAdd(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ))))),
    Transpose(Inverse(Inverse(MatMul(
        B, E.I, C.T, Transpose(Inverse(R))
    )))),
    Inverse(Inverse(MatMul(
        B, Transpose(Inverse(A)), R
    )))
)

testCase(algo = factor, expr = expr_GAaGM, check = check, byType = Transpose)
#assert expr_GAaGM.doit() == res.doit()
#assert res.doit() == check.doit()


# %% --------------------------------------------------------------


# TEST 13a: GA * GM = group symbol Add * group symbol Mul

expr_GAmGM = MatMul(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), C.T
    )))
)

check = MatMul(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Transpose(Inverse(MatMul(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

testCase(algo = factor, expr = expr_GAmGM, check = check, byType = Transpose)
#assert expr_GAmGM.doit() == res.doit()
#assert res.doit() == check.doit()
# %% --------------------------------------------------------------


# TEST 13b: GA * GM = group symbol Add * group symbol Mul (with more layerings per component)

expr_GAmGM = MatMul(
    Inverse(Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
        Inverse(Inverse(B)),
        Transpose(Transpose(C)),
        D.I
    ))))))),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatMul(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) ),
    Inverse(Transpose(Inverse(Transpose(MatMul(
        B, Inverse(Transpose(Transpose(Transpose(A)))), R
    )))))
)

check = MatMul(
    Transpose(Inverse(Inverse(Inverse(MatAdd(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ))))),
    Transpose(Inverse(Inverse(MatMul(
        B, E.I, C.T, Transpose(Inverse(R))
    )))),
    Inverse(Inverse(MatMul(
        B, Transpose(Inverse(A)), R
    )))
)

testCase(algo = factor, expr = expr_GAmGM, check = check, byType = Transpose)
#assert expr_GAmGM.doit() == res.doit()
#assert res.doit() == check.doit()


# %% --------------------------------------------------------------


# TEST 14a: GM + GM = group symbol Mul + group symbol Mul

expr_GMaGM = MatAdd(
    Inverse(MatMul(A.T, B.I, C, D.T)),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), C.T
    )))
)

check = MatAdd(
    Inverse(MatMul(A.T, B.I, C, D.T)),
    Transpose(Inverse(MatMul(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

testCase(algo = factor, expr = expr_GMaGM, check = check, byType = Transpose)
#assert expr_GMaGM.doit() == res.doit()
#assert res.doit() == check.doit()
# %% --------------------------------------------------------------


# TEST 14b: GM +  GM = group symbol Mul + group symbol Mul (with more layerings per component)

expr_GMaGM = MatAdd(
    Inverse(Inverse(Transpose(Inverse(Transpose(Transpose(MatMul(
        Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
        Inverse(Inverse(B)),
        Transpose(Transpose(C)),
        D.I
    ))))))),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatMul(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) ),
    Inverse(Transpose(Inverse(Transpose(MatMul(
        B, Inverse(Transpose(Transpose(Transpose(A)))), R
    )))))
)

check = MatAdd(
    Transpose(Inverse(Inverse(Inverse(MatMul(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ))))),
    Transpose(Inverse(Inverse(MatMul(
        B, E.I, C.T, Transpose(Inverse(R))
    )))),
    Inverse(Inverse(MatMul(
        B, Transpose(Inverse(A)), R
    )))
)

testCase(algo = factor, expr = expr_GMaGM, check = check, byType = Transpose)
#assert expr_GMaGM.doit() == res.doit()
#assert res.doit() == check.doit()


# %% --------------------------------------------------------------


# TEST 15a: GM * GM = group symbol Mul * group symbol Mul

expr_GMmGM = MatMul(
    Inverse(MatMul(A.T, B.I, C, D.T)),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), C.T
    )))
)

check = MatMul(
    Inverse(MatMul(A.T, B.I, C, D.T)),
    Transpose(Inverse(MatMul(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

testCase(algo = factor, expr = expr_GMmGM, check = check, byType = Transpose)
#assert expr_GMmGM.doit() == res.doit()
#assert res.doit() == check.doit()


# %% --------------------------------------------------------------


# TEST 15b: GM * GM = group symbol Mul * group symbol Mul

expr_GMmGM = MatMul(
    Transpose(MatMul(A, B)),
    Transpose(MatMul(R, J))
)

check = expr_GMmGM

testCase(algo = factor, expr = expr_GMmGM, check = check, byType = Transpose)
#assert expr_GMmGM.doit() == res.doit()
#assert res.doit() == check.doit()


# %% --------------------------------------------------------------


# TEST 15c: GM * GM = group symbol Mul * group symbol Mul (with more layerings per component)

expr_GMmGM = MatMul(
    Inverse(Inverse(Transpose(Inverse(Transpose(Transpose(MatMul(
        Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
        Inverse(Inverse(B)),
        Transpose(Transpose(C)),
        D.I
    ))))))),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatMul(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) ),
    Inverse(Transpose(Inverse(Transpose(MatMul(
        B, Inverse(Transpose(Transpose(Transpose(A)))), R
    )))))
)

check = MatMul(
    Transpose(Inverse(Inverse(Inverse(MatMul(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ))))),
    Transpose(Inverse(Inverse(MatMul(
        B, E.I, C.T, Transpose(Inverse(R))
    )))),
    Inverse(Inverse(MatMul(
        B, Transpose(Inverse(A)), R
    )))
)

testCase(algo = factor, expr = expr_GMmGM, check = check, byType = Transpose)
#assert expr_GMmGM.doit() == res.doit()
#assert res.doit() == check.doit()


# %%

#TODO later: SL_a_GA_2 as innermost arg layered with transpse and then combine with any other kind: SL, SA, SM, GA, GM in between the layers. TODO for every kind of expression

# TODO 2: must take all above tests and organize so each function I made gets all those above cases (then separate the "special" cases per function beneath these "general" cases which are made just by combining types of expressions, as opposed to the "biased"/"special" cases.)


# %% -------------------------------------------------------------




# GENERAL TEST 1: inverse out, transpose in
expr = Inverse(Transpose(C*E*B))

# %%
check = expr

testCase(algo = group, expr = expr, check = check, byType = Transpose)
# %%


check = Transpose(Inverse(MatMul(C, E, B)))

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)

# %%

# TODO should I leave the inverse / leave Constructors order as they are? Or is it ok for factor to act as rippleOut? 
check = Transpose(Inverse(MatMul(C, E, B)))

testCase(algo = factor, expr = expr, check = check, byType = Transpose)


# %%

check = Transpose(Inverse(MatMul(C, E, B)))

testCase(algo = polarize, expr = expr, check = check, byType = Transpose)

# %% -------------------------------------------------------------


# GENERAL TEST 2: transpose out, inverse in
expr = Transpose(Inverse(C*E*B))

# %%


check = expr

testCase(algo = group, expr = expr, check = check, byType = Transpose)

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)

testCase(algo = factor, expr = expr, check = check, byType = Transpose)

testCase(algo = polarize, expr = expr, check = check, byType = Transpose)

# %% -------------------------------------------------------------


# GENERAL TEST 3: individual transposes inside inverse
expr = Inverse(B.T * E.T * C.T)
# %%


check = Inverse(Transpose(C*E*B))

testCase(algo = group, expr = expr, check = check, byType = Transpose)
# %%


check = expr

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%

check = expr

testCase(algo = factor, expr = expr, check = check, byType = Transpose)
# %%

check = Transpose(Inverse(MatMul(C, E, B)))

testCase(algo = polarize, expr = expr, check = check, byType = Transpose)
# %% -------------------------------------------------------------


# GENERAL TEST 4: individual inverse inside transpose

expr = Transpose(B.I * E.I * C.I)
# %%

check = expr 

testCase(algo = group, expr = expr, check = check, byType = Transpose)

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)

testCase(algo = factor, expr = expr, check = check, byType = Transpose)

testCase(algo = polarize, expr = expr, check = check, byType = Transpose)


# %% -------------------------------------------------------------


# GENERAL TEST 5 a: individual symbols

Q = MatrixSymbol("Q", a, b)

# %%

(expr1, check1) = (Q, Q)
(expr2, check2) = (Q.T, Q.T)
(expr3, check3) = (C.I, C.I)
(expr4, check4) = (Inverse(Transpose(C)), Inverse(Transpose(C)))

testCase(algo = group, expr = expr1, check = check1, byType = Transpose)
testCase(algo = group, expr = expr2, check = check2, byType = Transpose)
testCase(algo = group, expr = expr3, check = check3, byType = Transpose)
testCase(algo = group, expr = expr4, check = check4, byType = Transpose)

# %%


(expr1, check1) = (Q, Q)
(expr2, check2) = (Q.T, Q.T)
(expr3, check3) = (C.I, C.I)
(expr4, check4) = (Inverse(Transpose(C)), Transpose(Inverse(C)))

testCase(algo = rippleOut, expr = expr1, check = check1, byType = Transpose)
testCase(algo = rippleOut, expr = expr2, check = check2, byType = Transpose)
testCase(algo = rippleOut, expr = expr3, check = check3, byType = Transpose)
testCase(algo = rippleOut, expr = expr4, check = check4, byType = Transpose)

# %%


(expr1, check1) = (Q, Q)
(expr2, check2) = (Q.T, Q.T)
(expr3, check3) = (C.I, C.I)
(expr4, check4) = (Inverse(Transpose(C)), Transpose(Inverse(C)))

testCase(algo = factor, expr = expr1, check = check1, byType = Transpose)
testCase(algo = factor, expr = expr2, check = check2, byType = Transpose)
testCase(algo = factor, expr = expr3, check = check3, byType = Transpose)
testCase(algo = factor, expr = expr4, check = check4, byType = Transpose)

# %%


(expr1, check1) = (Q, Q)
(expr2, check2) = (Q.T, Q.T)
(expr3, check3) = (C.I, C.I)
(expr4, check4) = (Inverse(Transpose(C)), Transpose(Inverse(C)))

testCase(algo = factor, expr = expr1, check = check1, byType = Transpose)
testCase(algo = factor, expr = expr2, check = check2, byType = Transpose)
testCase(algo = factor, expr = expr3, check = check3, byType = Transpose)
testCase(algo = factor, expr = expr4, check = check4, byType = Transpose)

# %%


(expr1, check1) = (Q, Q)
(expr2, check2) = (Q.T, Q.T)
(expr3, check3) = (C.I, C.I)
(expr4, check4) = (Inverse(Transpose(C)), Transpose(Inverse(C)))

testCase(algo = polarize, expr = expr1, check = check1, byType = Transpose)
testCase(algo = polarize, expr = expr2, check = check2, byType = Transpose)
testCase(algo = polarize, expr = expr3, check = check3, byType = Transpose)
testCase(algo = polarize, expr = expr4, check = check4, byType = Transpose)

# %% -------------------------------------------------------------



# GENERAL TEST 5 b: inidivudal symbols nested

expr = Transpose(Inverse(Inverse(Inverse(Transpose(MatMul(
    A,
    Inverse(Transpose(Inverse(Transpose(C)))),
    Inverse(Transpose(R)),
    M
))))))
# %%

check = Transpose(Inverse(Inverse(Inverse(Transpose(Transpose(MatMul(
    M.T,
    Transpose(Inverse(Transpose(R))),
    Transpose(Inverse(Transpose(Inverse(Transpose(C))))),
    A.T
)))))))

testCase(algo = group, expr = expr, check = check, byType = Transpose)

# %%


check = Transpose(Transpose(Inverse(Inverse(Inverse(MatMul(
    A,
    Transpose(Transpose(Inverse(Inverse(C)))),
    Transpose(Inverse(R)),
    M
))))))

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)

# %%


check = Inverse(Inverse(Inverse(MatMul(
    A, Inverse(Inverse(C)), Transpose(Inverse(R)), M
))))

testCase(algo = factor, expr = expr, check = check, byType = Transpose)
# %%


check = Inverse(Inverse(Inverse(MatMul(
    A, Inverse(Inverse(C)), Transpose(Inverse(R)), M
))))

testCase(algo = polarize, expr = expr, check = check, byType = Transpose)
# %% -------------------------------------------------------------



# GENERAL TEST 6: grouped products

expr = MatMul( Transpose(A*B), Transpose(R*J) )
# %%

check = Transpose(R*J*A*B)

testCase(algo = group, expr = expr, check = check, byType = Transpose)

# %%

check = expr

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%

check = expr 

testCase(algo = factor, expr = expr, check = check, byType = Transpose)
# %%


check = Transpose(MatMul(R, J, A, B))

testCase(algo = polarize, expr = expr, check = check, byType = Transpose)
# %% -------------------------------------------------------------


# GENERAL TEST 7: individual transposes littered along as matmul

expr = B.T * A.T * J.T * R.T
# %%


check = Transpose(R*J*A*B)

testCase(algo = group, expr = expr, check = check, byType = Transpose)
# %%

check = expr

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%

check = expr 

testCase(algo = factor, expr = expr, check = check, byType = Transpose)
# %%

check = Transpose(MatMul(R, J, A, B))

testCase(algo = polarize, expr = expr, check = check, byType = Transpose)
# %% -------------------------------------------------------------


# GENERAL TEST 8: inverses mixed with transpose in expr matmul, but with transposes all as the outer expression
L = Transpose(Inverse(MatMul(B, A, R)))

expr = MatMul(A , Transpose(Inverse(R)), Transpose(Inverse(L)) , K , E.T , B.I )
# %%

check = Transpose(
    MatMul( Transpose(Inverse(B)), E , K.T , Inverse(L) , Inverse(R) , A.T)
)

testCase(algo = group, expr = expr, check = check, byType = Transpose)
# %%

check = expr

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%

check = MatMul(
    A, Transpose(Inverse(R)), Inverse(Inverse(MatMul(B, A, R))), K, E.T, B.I
)

testCase(algo = factor, expr = expr, check = check, byType = Transpose)
# %%


check = MatMul(
    A, Transpose(Inverse(R)), Inverse(Inverse(MatMul(B, A, R))), K, E.T, B.I
)
testCase(algo = polarize, expr = expr, check = check, byType = Transpose)
# %% -------------------------------------------------------------


# GENERAL TEST 9: mix of inverses and transposes in expr matmul, but this time with transpose not as outer operation, for at least one symbol case.
L = Inverse(Transpose(MatMul(B, A, R)))

expr = MatMul(A , Transpose(Inverse(R)), Inverse(Transpose(L)) , K , E.T , B.I )
# %%


check = Transpose(
    MatMul( Transpose(Inverse(B)), E , K.T , Transpose(Inverse(Transpose(L))) , R.I , A.T)
)
testCase(algo = group, expr = expr, check = check, byType = Transpose)

# %%


check = MatMul(
    A, Transpose(Inverse(R)), Transpose(Transpose(Inverse(Inverse(MatMul(B, A, R))))), K, E.T, B.I
)

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%


check = MatMul(
    A, Transpose(Inverse(R)), Inverse(Inverse(MatMul(B, A, R))), K, E.T , B.I
)
testCase(algo = factor, expr = expr, check = check, byType = Transpose)
# %%


check = MatMul(
    A, Transpose(Inverse(R)), Inverse(Inverse(MatMul(B, A, R))), K, E.T , B.I
)
testCase(algo = polarize, expr = expr, check = check, byType = Transpose)

# %% -------------------------------------------------------------


# GENERAL TEST 10: transposes in matmuls and singular matrix symbols, all in expr matadd expression.
L = Transpose(Inverse(MatMul(B, A, R)))

expr = MatAdd( 
    MatMul(A, R.T, L, K, E.T, B),
    D.T, K
)
# %%


#check = MatAdd( Transpose(MatMul(B.T, E, K.T, Transpose(L), R, A.T)) , D.T, K)
check = MatAdd(
    Transpose(MatMul(B.T, E, K.T, Inverse(MatMul(B, A, R)), R, A.T)), K, D.T
)

testCase(algo = group, expr = expr, check = check, byType = Transpose)

# %%

check = Transpose(MatAdd(
    MatMul(B.T, E, K.T, Inverse(MatMul(B, A, R)), R, A.T), D, K.T
))

testGroupCombineAdds(expr = expr, check = check, byType = Transpose)

# %%


check = MatAdd(
    MatMul(A, R.T, Transpose(Inverse(MatMul(B, A, R))), K, E.T, B), 
    K, D.T
)

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)
testCase(algo = factor, expr = expr, check = check, byType = Transpose)
# %%


check = Transpose(MatAdd(
    D, MatMul(B.T, E, K.T, Inverse(MatMul(B, A, R)), R, A.T), K.T
))
testCase(algo = polarize, expr = expr, check = check, byType = Transpose)
# %% -------------------------------------------------------------


# GENERAL TEST 11: digger case, very layered expression (with transposes separated so not expecting the grouptranspose to change them). Has inner arg matmul.

expr = Trace(Transpose(Inverse(Transpose(C*D*E))))
# %%

check = expr

testCase(algo = group, expr = expr, check = check, byType = Transpose)

testGroupCombineAdds(expr = expr, check = check, byType = Transpose)

# %%

check = Trace(Transpose(Transpose(Inverse(MatMul(C, D, E)))))

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%



check = Trace(Inverse(MatMul(C, D, E)))

testCase(algo = factor, expr = expr, check = check, byType = Transpose)

testCase(algo = polarize, expr = expr, check = check, byType = Transpose)

# %% -------------------------------------------------------------



# GENERAL TEST 12: very layered expression, but transposes are next to each other, with inner arg as matmul

expr = Trace(Transpose(Transpose(Inverse(C*D*E))))
# %%

check = expr

testCase(algo = group, expr = expr, check = check, byType = Transpose)

testGroupCombineAdds(expr = expr, check = check, byType = Transpose)

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%

check = Trace(Inverse(MatMul(C, D, E)))

testCase(algo = factor, expr = expr, check = check, byType = Transpose)

testCase(algo = polarize, expr = expr, check = check, byType = Transpose)

# %% -------------------------------------------------------------


# GENERAL TEST 13: very layered expression (digger case) with individual transpose and inverses littered in the inner matmul arg.
expr = Transpose(Inverse(Transpose(C.T * A.I * D.T)))
# %%


check = Transpose(Inverse(Transpose(Transpose(
    MatMul(D , Transpose(Inverse(A)), C )
))))

testCase(algo = group, expr = expr, check = check, byType = Transpose)
# %%

check = Transpose(Transpose(Inverse(MatMul(C.T, A.I, D.T))))

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%


check = Inverse(C.T * A.I * D.T)

testCase(algo = factor, expr = expr, check = check, byType = Transpose)
testCase(algo = polarize, expr = expr, check = check, byType = Transpose)

# %% -------------------------------------------------------------



# GENERAL TEST 14 a: Layered expression (many innermost nestings). The BAR expression has transpose outer and inverse inner, while the other transposes are outer already. (ITIT)

L = Transpose(Inverse(B*A*R))

expr = Transpose(Inverse(
    MatMul(A*D.T*R.T ,
        Transpose(
            Inverse(
                MatMul(C.T, E.I, L, B.T)
            )
        )
    )
))

# %%

check = Transpose(
    Inverse(Transpose(
        MatMul(
            Inverse(Transpose(MatMul(
                B, Inverse(B*A*R), Transpose(Inverse(E)), C
            ))),

            Transpose(Transpose(MatMul(R, D, A.T)))
        )
    ))
)

testCase(algo = group, expr = expr, check = check, byType = Transpose)
testGroupCombineAdds(expr = expr, check = check, byType = Transpose)
# %%

check = expr 

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)

testCase(algo = factor, expr = expr, check = check, byType = Transpose)
# %%

check = Inverse(MatMul(
    Transpose(Inverse(MatMul(
        B, Inverse(B*A*R), Transpose(Inverse(E)), C
    ))), 
    MatMul(R, D, A.T)
))

testCase(algo = polarize, expr = expr, check = check, byType = Transpose)


# %% -------------------------------------------------------------




# GENERAL TEST 14 b: layered expression (many innermost nestings). The BAR expression has transpose inner and inverse outer and other transposes are outer already, after inverse. (ITIT)

L = Inverse(Transpose(B*A*R))

expr = Transpose(Inverse(
    MatMul(A*D.T*R.T ,
        Transpose(
            Inverse(
                MatMul(C.T, E.I, L, B.T)
            )
        )
    )
))

# %%

check = Transpose(Inverse(Transpose(MatMul(
    Inverse(Transpose(MatMul(
        B, Transpose(Inverse(Transpose(B*A*R))), Transpose(Inverse(E)), C
    ))),
    Transpose(Transpose(MatMul(R, D, A.T)))
))))

testCase(algo = group, expr = expr, check = check, byType = Transpose)
# %%

check = Transpose(Inverse(MatMul(
    A, D.T, R.T, 
    Transpose(Inverse(MatMul(
        C.T, E.I, Transpose(Inverse(MatMul(B, A, R))), B.T
    )))
)))

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)

testCase(algo = factor, expr = expr, check = check, byType = Transpose)
# %%


check = Inverse(MatMul(
    Transpose(Inverse(MatMul(
        B, Inverse(B*A*R), Transpose(Inverse(E)), C
    ))),
    MatMul(R, D, A.T)
))

testCase(algo = polarize, expr = expr, check = check, byType = Transpose)

# %% -------------------------------------------------------------


# GENERAL TEST 14 c: innermost layered expression, with BAR having tnraspose outer and inverse inner, and the other expressions have transpose inner and inverse outer (TITI)
L = Transpose(Inverse(B*A*R))

expr = Inverse(Transpose(
    MatMul(A*D.T*R.T ,
        Inverse(
            Transpose(
                MatMul(C.T, E.I, L, B.T)
            )
        )
    )
))

# %%


check = Inverse(Transpose(Transpose(MatMul(
    Transpose(Inverse(Transpose(Transpose(MatMul(
        B, Inverse(B*A*R), Transpose(Inverse(E)), C
    ))))),
    Transpose(Transpose(MatMul(R, D, A.T)))
))))

testCase(algo = group, expr = expr, check = check, byType = Transpose)

# %%

check = Transpose(Inverse(MatMul(
    A, D.T, R.T,
    Transpose(Inverse(MatMul(
        C.T, E.I, Transpose(Inverse(B*A*R)), B.T
    )))
)))

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)

testCase(algo = factor, expr = expr, check = check, byType = Transpose)
# %%


check = Inverse(MatMul(
    Transpose(Inverse(MatMul(
        B, Inverse(B*A*R), Transpose(Inverse(E)), C
    ))),
    MatMul(R, D, A.T)
))

testCase(algo = polarize, expr = expr, check = check, byType = Transpose)
# %% -------------------------------------------------------------



# GENERAL TEST 14 d: innermost layered expression, with BAR having tnraspose inner and inverse outer, and the other expressions have transpose inner and inverse outer (TITI)

L = Inverse(Transpose(B*A*R))

expr = Inverse(Transpose(
    MatMul(A*D.T*R.T ,
        Inverse(
            Transpose(
                MatMul(C.T, E.I, L, B.T)
            )
        )
    )
))

# %%

check = Inverse(Transpose(Transpose(MatMul(
    Transpose(Inverse(Transpose(Transpose(MatMul(
        B, Transpose(Inverse(Transpose(B*A*R))), Transpose(Inverse(E)), C
    ))))),
    Transpose(Transpose(MatMul(R, D, A.T)))
))))

testCase(algo = group, expr = expr, check = check, byType = Transpose)
# %%


check = Transpose(Inverse(MatMul(
    A, D.T, R.T,
    Transpose(Inverse(MatMul(
        C.T, E.I, Transpose(Inverse(B*A*R)), B.T
    )))
)))

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)
testCase(algo = factor, expr = expr, check = check, byType = Transpose)
# %%

check = Inverse(MatMul(
    Transpose(Inverse(MatMul(
        B, Inverse(B*A*R), Transpose(Inverse(E)), C
    ))),
    MatMul(R, D, A.T)
))
testCase(algo = polarize, expr = expr, check = check, byType = Transpose)
# %% -------------------------------------------------------------



# GENERAL TEST 15: very layered expression, and the inner arg is matadd with some matmuls but some matmuls have one of the elements as another layered arg (any of the test 14 cases, like 14 expr), so we can test if the function reaches all rabbit holes effectively.

L = Transpose(Inverse(B*A*R))

expr_inv_14a = Inverse(
    MatMul(A*D.T*R.T ,
        Transpose(
            Inverse(
                MatMul(C.T, E.I, L, B.T)
            )
        )
    )
)

expr = Transpose(
    MatAdd( B.T * A.T * J.T * R.T,  expr_inv_14a
    )
)
# %%



check_inv_14a = Inverse(Transpose(
        MatMul(
            Inverse(Transpose(MatMul(
                B, Inverse(B*A*R), Transpose(Inverse(E)), C
            ))),

            Transpose(Transpose(MatMul(R, D, A.T)))
        )
))

check = Transpose(
    MatAdd(
        check_inv_14a,

        Transpose(MatMul(R, J, A, B))
    )
)

testCase(algo = group, expr = expr, check = check, byType = Transpose)
# %%


check_inv_14a = Inverse(Transpose(
        MatMul(
            Inverse(Transpose(MatMul(
                B, Inverse(B*A*R), Transpose(Inverse(E)), C
            ))),

            Transpose(Transpose(MatMul(R, D, A.T)))
        )
))

check = Transpose(Transpose(
    MatAdd(
        Transpose(check_inv_14a),

        MatMul(R, J, A, B)
    )
))

testGroupCombineAdds(expr = expr, check = check, byType = Transpose)
# TODO: faulty function: group combine add should have gone in the inside of the outer transpose and done transpose combine there. 

# %%


L = Inverse(Transpose(B*A*R))

expr_inv_14d = Inverse(
    MatMul(A*D.T*R.T ,
        Inverse(
            Transpose(
                MatMul(C.T, E.I, L, B.T)
            )
        )
    )
)

expr = Transpose(
    MatAdd( B.T * A.T * J.T * R.T,  expr_inv_14d )
)

# Simple check 14 expr
check_inner_14d = Inverse(
    MatMul(A*D.T*R.T ,
        Transpose(
            Inverse(
                MatMul(C.T, E.I, Transpose(Inverse(B*A*R)), B.T)
            )
        )
    )
)

check = Transpose(MatAdd(
    B.T * A.T * J.T * R.T,
    check_inner_14d
))

testCase(algo = rippleOut, expr = expr, check = check, byType = Transpose)
testCase(algo = factor, expr = expr, check = check, byType = Transpose)

# %%


L = Inverse(Transpose(B*A*R))

expr_inv_14d = Inverse(
    MatMul(A*D.T*R.T ,
        Inverse(
            Transpose(
                MatMul(C.T, E.I, L, B.T)
            )
        )
    )
)

expr = Transpose(
    MatAdd( B.T * A.T * J.T * R.T,  expr_inv_14d )
)

check = MatAdd(
    Inverse(MatMul(
        Transpose(Inverse(MatMul(
            B, Inverse(B*A*R), Transpose(Inverse(E)), C
        ))), 
        MatMul(R, D, A.T)
    )),
    MatMul(R, J, A, B)
)

testCase(algo = polarize, expr = expr, check = check, byType = Transpose)

# %%


# GENERAL TEST 16: testing mix and match of matmul / matadd with inverse / transposes to see how polarize filters out Transpose. (Meant for mainly testing the polarize function)

expr_polarize = Inverse(MatMul(
    Transpose(Inverse(Transpose(MatAdd(B.T, A.T, R, MatMul(Transpose(Inverse(B*A*R.T)), MatAdd(E, J, D)), Inverse(Transpose(E)), Inverse(Transpose(D)))))),
    Inverse(Transpose(MatMul(A.T, B.T, E.I, Transpose(Inverse(Transpose(A + E + R.T))), C)))
))
# %%


check = Inverse(Transpose(MatMul(
    Transpose(Inverse(Transpose(Transpose(MatMul(
        C.T, Inverse(Transpose(A + E + R.T)), Transpose(Inverse(E)), B, A
    ))))), 
    Inverse(Transpose(MatAdd(
        Inverse(Transpose(D)), 
        Inverse(Transpose(E)), 
        R, 
        Transpose(MatMul(
            Transpose(D + E + J), 
            Inverse(Transpose(R * A.T * B.T))
        )),
        A.T, 
        B.T
    )))
)))

testCase(algo = group, expr = expr_polarize, check = check, byType = Transpose)
# %%


check = Inverse(MatMul(
    Inverse(MatAdd(
        B.T, A.T, R, MatMul(
            Transpose(Inverse(B*A*R.T)), 
            MatAdd(E, J, D)
        ), 
        Transpose(Inverse(E)), 
        Transpose(Inverse(D))
    )),
    Transpose(Inverse(
        MatMul(
            A.T, B.T, E.I, 
            Inverse(A + E + R.T), 
            C
        )
    ))
))

testCase(algo = factor, expr = expr_polarize, check = check, byType = Transpose)
# %%


check = Transpose(Inverse(MatMul(
    Inverse(MatMul(
        A.T, B.T, E.I, Inverse(MatAdd(A, E, R.T)), C
    )), 
    Inverse(MatAdd(
        D.I, E.I, A, B, MatMul(
            Transpose(MatAdd(D, E, J)), 
            Inverse(MatMul(B, A, R.T))
        ), 
        R.T
    ))
)))

testCase(algo = polarize, expr = expr_polarize, check = check, byType = Transpose)
# %%




### TRACE DERIVATIVE HERE ----------------------------------------


# Now apply the trace deriv function per pair
def traceDerivPair(pair: Tuple[List[MatrixSymbol], List[MatrixSymbol]]) -> MatrixExpr:
    (left, right) = pair

    # NOTE: separating operations based on the length of the element list because we don't want braces like (C)^T, instead we want C^T simply. If len == 1 then it is case C instaed of C,E,B,,... so need no MatMul inside Transpose:
    leftTransp = Transpose(*left) if len(left) == 1 else Transpose(MatMul(*left))

    rightTransp = Transpose(*right) if len(right) == 1 else Transpose(MatMul(*right))

    return MatMul( leftTransp, rightTransp )

# %%
def derivMatInsideTrace(expr: MatrixExpr, byVar: MatrixSymbol) -> MatMul:
    # First check if not expr matrix symbol; if it is then diff easily.
    if isinstance(expr, MatrixSymbol):
        return diff(Trace(expr), byVar)

    # Check now that it is matmul
    assert expr.is_MatMul

    # Get how many of byVar are in the expression:
    # NOTE: arg may be under Transpose or Inverse here, not just expr simple MatrixSymbol, so we need to detect the MatrixSymbol underneath using "has" test instead of the usual == test of arg == byVar.
    # This way we count the byVar underneath the Transp or Inverse.
    numSignalVars = len(list(filter(lambda arg: arg.has(byVar), expr.args)))
    # NOTE here is expr way to get the underlying MatrixSymbol when there are Invs and Transposes (just the letters without the invs and transposes):
    # list(itertools.chain(*map(lambda expr: expr.free_symbols, trace.arg.args))), where expr := trace.arg

    # Get the split list applications: split by signal var for how many times it appears
    signalSplits = list(map(lambda n: splitOnce(expr.args, byVar, n), range(1, numSignalVars + 1)))

    # Apply the trace derivative function per pair
    transposedMatMuls = list(map(lambda s: traceDerivPair(s), signalSplits))

    # Result is an addition of the transposed matmul combinations:
    return MatAdd(* transposedMatMuls )



# %%



def derivTrace(trace: Trace, byVar: MatrixSymbol) -> MatrixExpr:
    '''
    Does derivative of expr Trace expression.
    Equivalent to diff(trace, byVar).
    '''
    assert trace.is_Trace

    # Case 1: trace of expr single matrix symbol - easy
    if isinstance(trace.arg, MatrixSymbol):
        return diff(trace, byVar)

    # Case 2: if arg is matmul then just apply the trace matmul function:
    elif trace.arg.is_MatMul:
        return derivMatInsideTrace(trace.arg, byVar = byVar)
        #assert equal(result, diff(trace, byVar))

    # Case 3: split first by MatAdd to get MatMul pieces and feed in the pieces to the single function that gets applied to each MatMul piece.
    elif trace.arg.is_MatAdd:
        # Filter the matrixsymbols that are byVar and the matrixexprs that contain the byVar
        addends: List[MatrixExpr] = list(filter(lambda m : m.has(byVar), trace.arg.args ))
        # NOTE: can contain matrixsymbols mixed with matmul

        # NOTE this is expr list of MatAdds; must flatten them to avoid brackets extra and to enhance simplification.
        diffedAddends: List[MatrixExpr] = list(map(lambda m : derivMatInsideTrace(m, byVar), addends))

        # Preparing to flatten the matrix additions into one overall matrix addition:
        splitMatAdd = lambda expr : list(expr.args) if expr.is_MatAdd else [expr]

        # Splitting and flattening here:
        splitDiffedAddends = list(itertools.chain(*map(lambda d : splitMatAdd(d), diffedAddends)) )

        # Now return the mat add
        return MatAdd(*splitDiffedAddends)





# %% -------------------------------------------------------------




### TEST 1: simple case, with addition, no inverse or transpose in any of the variables
a, b, c = symbols('expr b c', commutative=True)

C = MatrixSymbol('C', a, b)
E = MatrixSymbol('E', b, c)
B = MatrixSymbol('B', c, b)
L = MatrixSymbol('L', c, b)
D = MatrixSymbol('D', c, a)

K = MatrixSymbol('K', a, a)


trace = Trace(expr + A*B*E * R * J  + A*D + K )
byVar = E


res = derivTrace(trace, byVar)

check = MatAdd(
    MatMul(Transpose(MatMul(A, B)), Transpose(MatMul(R, J))),
    MatMul(C.T, Transpose(MatMul(B, E, L, E, D))),
    MatMul(Transpose(MatMul(C, E, B)), Transpose(MatMul(L, E, D))),
    MatMul(Transpose(MatMul(C, E, B, E, L)), D.T)
)

dcheck = diff(trace, byVar)

# NOTE: doesn't work to simplify the check - res expression ! Leaves it in subtraction form, same with doit() in all kinds of combinations with simplify()
#assert simplify(check - res) == 0

assert equal(check, res)
assert equal(res, dcheck)
assert equal(check, dcheck)

showGroup([
    res,
    group(res),
    dcheck
])


# %% -------------------------------------------------------------


# TEST 2a: testing one inverse expression, not the byVar

C = MatrixSymbol('C', a, c)
E = MatrixSymbol('E', c, c)
B = MatrixSymbol('B', c, c)
L = MatrixSymbol('L', c, c)
D = MatrixSymbol('D', c, a)

trace = Trace(C * E * B * E * Inverse(L) * E * D)
byVar = E

res = derivTrace( trace , byVar)

check = Transpose( B * E * Inverse(L) * E * D * C + Inverse(L)*E*D*C*E*B +  D*C*E*B*E*Inverse(L))

dcheck = diff(trace, byVar)

assert equal(res, check)
assert equal(res, dcheck)
assert equal(check, dcheck)

showGroup([
    res,
    check,
    dcheck
])

# %% -------------------------------------------------------------


# TEST 2b: testing one inverse expression, not the byVar

C = MatrixSymbol('C', c, c)
A = MatrixSymbol('A', c, c)
E = MatrixSymbol('E', c, c)
B = MatrixSymbol('B', c, c)
L = MatrixSymbol('L', c, c)
D = MatrixSymbol('D', c, c)

trace = Trace(B * Inverse(C) * E * L * A * E * D)
byVar = E

res = derivTrace( trace , byVar)

check = Transpose(L * A* E * D * B * Inverse(C) + D*B*Inverse(C)*E*L*A)

dcheck = diff(trace, byVar)

assert equal(res, check)
assert equal(res, dcheck)
assert equal(check, dcheck)

showGroup([
    res,
    check,
    dcheck
])

# %% -------------------------------------------------------------


# TEST 2c: testing one inverse expression, not the byVar, that is situated at the front of the expression.

C = MatrixSymbol('C', c, c)
E = MatrixSymbol('E', c, b)
B = MatrixSymbol('B', b, c)
L = MatrixSymbol('L', b, c)
D = MatrixSymbol('D', b, c)

trace = Trace(Inverse(C) * E * B * E * L * E * D)
byVar = E

res = derivTrace( trace , byVar)

check = Transpose(B*E*L*E*D*Inverse(C) + L*E*D*Inverse(C)*E*B + D*Inverse(C)*E*B*E*L)

dcheck = diff(trace, byVar)

assert equal(res, check)
assert equal(res, dcheck)
assert equal(check, dcheck)

showGroup([
    res,
    check,
    dcheck
])

# %% -------------------------------------------------------------


# TEST 3a: testing mix of inverse and transpose expressions, not the byVar

B = MatrixSymbol('B', c, c)
C = MatrixSymbol('C', c, c)
E = MatrixSymbol('E', c, c)
L = MatrixSymbol('L', b, c)
A = MatrixSymbol('A', b, c)
D = MatrixSymbol('D', c, c)

trace = Trace(B * C.I * E * L.T * A * E * D)
byVar = E

res = derivTrace( trace , byVar)

check = Transpose(L.T * A * E * D * B * C.I) + Transpose(D *B*C.I*E*L.T*A)

dcheck = diff(trace, byVar)

assert equal(res, check)
assert equal(res, dcheck)
assert equal(check, dcheck)

showGroup([
    res,
    check,
    dcheck,
    group(res)
])
# %% -------------------------------------------------------------



### TEST 3b: testing mix of inverser and transpose expressions, and byVar is either an inverse of transpose.

B = MatrixSymbol('B', c, c)
C = MatrixSymbol('C', c, c)
E = MatrixSymbol('E', c, c)
L = MatrixSymbol('L', b, c)
A = MatrixSymbol('A', b, c)
D = MatrixSymbol('D', c, c)

trace = Trace(B * C.I * E.T * L.T * A * E * D)
byVar = E

res = derivTrace( trace , byVar)

check = L.T * A*E*D*B*C.I + A.T* L * E *(C.I).T * B.T * D.T

dcheck = diff(trace, byVar)

showGroup([
    res, check, dcheck, group(dcheck)
])
# %%
assert equal(res, check)
assert equal(res, dcheck)
assert equal(check, dcheck)

showGroup([
    res,
    check,
    dcheck,
    group(res)
])


# %%
# TODO fix this function so it can take symbolic matrices
#diffMatrix(B * Inverse(C) * E.T * L.T * A * E * D,   E)

#diffMatrix(B_ * Inverse(C_) * E_.T * L_.T * A_ * E_ * D_,   E_)

# TODO this function seems very wrong: just seems to add the differential operator to the byVar instead of actually doing anything to the expression.

# TODO: this function's result doesn't match the derivMatmul result of this expression


# %%
#TODO gives error
# matrixDifferential(X, X)
# %%
# matrixDifferential(expr*X, X)
# %%
# TODO error with cyclic permute
# # matrixDifferential(Trace(R_), R_)
# %%
# matrixDifferential(A*J, A)

