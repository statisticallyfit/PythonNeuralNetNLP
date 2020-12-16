
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

from src.NeuralNetworkStudy.books.SethWeidman_DeepLearningFromScratch.FunctionUtil import *


from src.utils.GeneralUtil import *

from src.MatrixCalculusStudy.MatrixDerivLib.symbols import Deriv
from src.MatrixCalculusStudy.MatrixDerivLib.diff import matrixDifferential
from src.MatrixCalculusStudy.MatrixDerivLib.printingLatex import myLatexPrinter

# For displaying
from IPython.display import display, Math
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax', latex_printer= lambda e, **kw: myLatexPrinter.doprint(e))





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
def splitOnce(theArgs: List[MatrixSymbol], signalVar: MatrixSymbol, n: int) -> Tuple[List[MatrixSymbol], List[MatrixSymbol]]:

    assert n <= len(theArgs) and abs(n) == n

    cumArgs = []
    countSignal: int = 0

    for i in range(0, len(theArgs)):
        arg = theArgs[i]

        if arg == signalVar:
            countSignal += 1

            if countSignal == n:
                return (cumArgs, list(theArgs[i + 1: ]) )


        cumArgs.append(arg)

    return ([], [])

# %%

# TESTING: splitOnce()

L = MatrixSymbol('L', c, c)

expr = C*E*B*E*L*E*D

assert splitOnce(expr.args, E, 1) == ([C], [B, E, L, E, D])
assert splitOnce(expr.args, E, 2) == ([C, E, B], [L, E, D])
assert splitOnce(expr.args, E, 3) == ([C, E, B, E, L], [D])
assert splitOnce(expr.args, E, 0) == ([], [])
assert splitOnce(expr.args, E, 4) == ([], [])
# TODO how to assert error for negative number n?



# %%
hasConstr = lambda Constr, expr : Constr.__name__ in srepr(expr)
#hasTranspose = lambda e : "Transpose" in srepr(e)

def pickOut(Constr: MatrixType, expr: MatrixExpr):
    #if not hasTranspose(expr):
    if not hasConstr(Constr, expr):
        #return Transpose(MatMul(*expr.args))
        #return Transpose(expr)
        return Constr(expr)
    elif hasConstr(Constr, expr) and expr.func == Constr:
    #elif hasTranspose(expr) and expr.is_Transpose:
        return expr.arg

    return Constr(expr) #TODO check

# %%
def algo_Group_MatMul_or_MatSym(byType: MatrixType, expr: MatrixExpr) -> MatrixExpr:

    def revAlgo(Constr: MatrixType, expr: MatrixExpr):
        '''Converts B.T * A.T --> (A * B).T'''

        if hasConstr(Constr, expr) and expr.is_MatMul:

            revs = list(map(lambda a : pickOut(a), reversed(expr.args)))

            #return Transpose(MatMul(*revs))
            return Constr(MatMul(*revs))

        return expr

    ps = list(preorder_traversal(expr))
    ms = list(filter(lambda p : p.is_MatMul, ps))
    ts = list(map(lambda m : revAlgo(m), ms))

    ns = []
    for j in range(0, len(ms)):
        m = ms[j]
        if expr.has(m):
            #newPair = [(m, ts[j])] #revTransposeAlgo(m))]
            #ds[expr] = (ds.get(expr) + newPair) if expr in ds.keys() else newPair
            ns.append( (m, ts[j]) )

    # Apply the changes in the list of replacements we gathered above
    exprToChange = expr
    for (old, new) in ns:
        exprToChange = exprToChange.xreplace({old : new})

    return exprToChange


# %% -------------------------------------------------------------


# TEST 1: more than one innermost expression
L = Transpose(Inverse(B*A*R))

expr = Transpose(Inverse(
    MatMul(A*D.T*R.T , Transpose(Inverse(
        MatMul(C.T, E.I, L, B.T)
    )
))))

check = Transpose(
    Inverse(Transpose(
        MatMul(
            Inverse(Transpose(MatMul(
                B, Inverse(MatMul(B, A, R)), Transpose(Inverse(E)), C
            ))),

            Transpose(Transpose(MatMul(R, D, A.T)))
        )
    ))
)

res = algo_Group_MatMul_or_MatSym(expr)

showGroup([
    expr,
    res,
    check,
    expr.doit(),
    res.doit()
])

assert equal(res, expr)
assert equal(res, check) # # TODO want to use == for structurally equal tests here

# %%
### GROUP TRANSPOSE HERE ----------------------------------------


def group(byType: MatrixType, expr: MatrixExpr, combineAdds:bool = False) -> MatrixExpr:
    '''Combines transposes when they are the outermost operations.

    Brings the individual transpose operations out over the entire group (factors out the transpose and puts it as the outer operation).
    NOTE: This happens only when the transpose ops are at the same level. If they are nested, that "bringing out" task is left to the rippletranspose function.

    combineAdds:bool  = if True then the function will group all the addition components under expr transpose, like in matmul, otherwise it will group only the individual components of the MatAdd under transpose.'''

    #isMatSym = lambda s: len(s.free_symbols) == 1

    if not expr.is_MatAdd: # TODO must change this to identify matmul or matsym exactly
        return algo_Group_MatMul_or_MatSym(expr)

    # TODO fix this to handle any non-add operation upon function entry
    addendsTransp: List[MatrixExpr] = list(map(lambda a: group(a), expr.args))

    if combineAdds:
        innerAddends = list(map(lambda t: pickOut(t), addendsTransp))
        return Transpose(MatAdd(*innerAddends))

    # Else not combining adds, just grouping transposes in addends individually
    # TODO fix this to handle any non-add operation upon function entry. May not be a MatAdd, may be Trace for instance.
    return MatAdd(*addendsTransp)


# %%




### TRANSPOSE OUT HERE -----------------------------------------

# Need to include list of constructors that you dig out of.
# TODO: need Trace / Derivative / Function ...??
CONSTR_LIST: List[MatrixType] = [Transpose, Inverse]


# TODO to add Function, Derivative, and Trace ? any others?
ALL_TYPES_LIST = [Transpose, Inverse, MatMul, MatAdd, MatrixSymbol, Symbol, Number]

# %%


def chunkTypesBy(byTypes: List[MatrixType], types: List[MatrixType]) -> List[List[MatrixType]]:
    '''Separates the `types` in expr list of list of types by the types in `byConstrs` and keeps other types separate too, just as how they appear in the original list'''

    # Not strictly necessary here, just useful if you want to see shorter version of names below
    #getSimpleTypeName = lambda t : str(t).split("'")[1].split(".")[-1]
    byConstrs: List[MatrixType] = list(set(byTypes))

    # Step 1: coding up in pairs for easy identification: need Inverse and Transpose tagged as same kind
    codeConstrPairs: List[Tuple[int, MatrixType]] = list(map(lambda c : (0, c) if (c in byConstrs) else (1, c), types))

    # Step 2: getting the groups
    chunkedPairs: List[List[Tuple[int, MatrixType]]] = [list(group) for key, group in itertools.groupby(codeConstrPairs, operator.itemgetter(0))]

    # Step 3: getting only the types in the chunked lists
    chunkedConstrs: List[List[MatrixType]] = list(map(lambda lst : list(map(lambda pair : pair[1], lst)), chunkedPairs))

    return chunkedConstrs
# %%

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
# %%

#clean = lambda t : str(t).split("'")[1].split(".")[-1]



def stackTypesBy(byType: MatrixType, types: List[MatrixType]) -> List[MatrixType]:
    '''Given expr type (like Transpose) and given expr list, this function pulls all the signal types to the end of the
    list, leaving the non-signal-types as the front'''

    # Get number of signal types in the list
    countTypes: int = len(list(filter(lambda c : c == byType, types)))

    # Create the signal types that go at the end
    endTypes: List[MatrixType] = [byType] * countTypes

    # Get the list without the signal types
    nonSignalTypes: List[MatrixType] = list(filter(lambda c : c != byType, types))

    return endTypes + nonSignalTypes # + endTypes

assert stackTypesBy(Transpose, [Inverse, Inverse, Transpose, Inverse, Transpose, Transpose, Inverse, MatMul, MatAdd]) == [Transpose, Transpose, Transpose, Inverse, Inverse, Inverse, Inverse, MatMul, MatAdd]



# %%



def inner(expr):
    '''Gets the innermost expression (past all the .arg) on the first level only'''
    #isMatSym = lambda e : len(e.free_symbols) == 1

    # TODO missing any base case possibilities? Should include here anything that is not Trace / Inverse ... etc or any kind of constructors that houses an inner argument.
    Constr = expr.func
    #types = [MatMul, MatAdd, MatrixSymbol, Symbol, Number]
    otherTypes = list( set(ALL_TYPES_LIST).symmetric_difference(CONSTR_LIST) )
    #isAnySubclass = any(map(lambda t : issubclass(Constr, t), types))

    if (Constr in otherTypes) or issubclass(Constr, Number):

        return expr

    # else keep recursing
    return inner(expr.arg) # need to get arg from Trace or Transpose or Inverse ... among other constructors


# %% -------------------------------------------------------------


# TEST 1: simplest case possible, just matrixsymbol as innermost nesting
t1 = Transpose(Inverse(Inverse(Inverse(Transpose(Transpose(A))))))
assert inner(t1) == A

# TEST 2: second nesting inside the innermost expression
t2 = Transpose(Inverse(Transpose(Inverse(Transpose(Inverse(Inverse(MatMul(B, A, Transpose(Inverse(A*C)) ) )))))))
c2 = MatMul(B, A, Transpose(Inverse(A*C)))

assert inner(t2) == c2

# TEST 3: testing the real purpose now: the inner() function must get just the first level of innermosts:
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



# %%

def applyTypesToExpr( pairTypeExpr: Tuple[List[MatrixType], MatrixExpr]) -> MatrixExpr:
    '''Ignores any types that are MatrixSymbol or Symbol or instance of Number because those give error when applied to an expr.'''

    (typeList, expr) = pairTypeExpr

    isSymOrNum = lambda tpe : tpe in [MatrixSymbol, Symbol] or issubclass(tpe, Number)

    typeList = list(filter(lambda tpe: not isSymOrNum(tpe), typeList))

    if typeList == []:
        return expr
    return compose(*typeList)(expr)

# %%

anyTypeIn = lambda testTypes, searchTypes : any([True for tpe in testTypes if tpe in searchTypes])


def chunkExprsBy(byTypes: List[MatrixType], expr: MatrixExpr) -> Tuple[List[List[MatrixType]], List[List[MatrixExpr]]]:
    '''Given an expression, returns the types and expressions as tuples, listed in preorder traversal, and separated by which expressions are grouped by Transpose or Inverse constructors (in their layering)'''

    byConstrs: List[MatrixType] = list(set(byTypes))

    ps: List[MatrixExpr] = list(preorder_traversal(expr)) # elements broken down
    cs: List[MatrixType] = list(map(lambda p: type(p), ps)) # types / constructors


    # Check first: does the expr have the types in byConstrs? If not, then return it out, nothing to do here:
    #if not (Transpose in cs):
    #BAD tests for AND all types: if not (set(byConstrs).intersection(cs) == set(byConstrs)):
    # GOOD, tests for OR all types (since should be able to chunk [T, T, T] then [I, I] and [I, T, I, T] so need the OR relationship):
    if not anyTypeIn(testTypes = CONSTR_LIST, searchTypes = cs):
        return ([cs], [ps])

    csChunked: List[List[MatrixType]] = chunkTypesBy(byTypes = byConstrs, types = cs)


    # Get the lengths of each chunk
    chunkLens: List[int] = list(map(lambda cLst : len(cLst), csChunked))


    # Use lengths to segregate the preorder traversal exprs also, then later to apply the transformations
    psChunked: List[List[MatrixExpr]] = []
    rest: List[MatrixExpr] = ps

    for size in chunkLens:
        (fst, rest) = (rest[:size], rest[size: ])
        psChunked.append( fst )

    return (csChunked, psChunked)
# %%




def rippleOut(byType: MatrixType, expr: MatrixExpr) -> MatrixExpr:

    '''Brings transposes to the outermost level when in expr nested expression. Leaves the nested expressions in their same structure.
    Because it preserves the nesting structure, it is expr kind of shallow polarize() function'''


    def algo_RippleOut_MatMul_or_MatSym(byType: MatrixType, expr: MatrixExpr) -> MatrixExpr:
        '''For each layered (nested) expression where transpose is the inner operation, this function brings transposes to be the outer operations, leaving all other operations in between in the same order.'''

        assert byType in CONSTR_LIST

        (csChunked, psChunked) = chunkExprsBy(byTypes = CONSTR_LIST, expr = expr)

        # Order the types properly now for each chunk: make transposes go last in each chunk:
        stackedChunks = list(map(lambda lst : stackTypesBy(byType = byType, types = lst), csChunked))

        # Pair up the correct order of transpose types with the expressions
        # BEFORE: (Transpose in tsPs[0]) or (Inverse in tsPs[0])
        typeListExprListPair = list(filter(lambda tsPs : anyTypeIn(testTypes = CONSTR_LIST, searchTypes = tsPs[0]),
                                           list(zip(stackedChunks, psChunked))))



        #typeListExprPair = list(map(lambda tsPs : (tsPs[0], tsPs[1][0]), typeListExprListPair))

        # Get the first expression only, since it is the most layered, don't use the entire expression list of the
        # tuple's second part. And get the inner argument (lay bare) in preparation for apply the correct order of
        # `byType`s.
        typeListInnerExprPair = list(map(lambda tsPs : (tsPs[0], inner(tsPs[1][0])), typeListExprListPair))

        outs = list(map(lambda tsExprPair : applyTypesToExpr(tsExprPair), typeListInnerExprPair ))

        # Get the original expressions as they were before applying correct transpose
        ins = list(map(lambda tsPs : tsPs[1][0], typeListExprListPair))

        # Filter: get just the matmul-type arguments (meaning not the D^T or E^-1 type arguments) from the result list (assuming there are other MatMul exprs). Could have done this when first filtering the psChunked, but easier to do it now.
        # NOTE: when there are ONLY mat syms and no other matmuls then we must keep them since it means the expression is layered with only expr matsum as the innermost expression, rather than expr matmul.
        isSymOrNum = lambda expr : expr.is_Symbol or expr.is_Number
        #isSym = lambda expr  : len(expr.free_symbols) == 1

        # Flattening the chunked ps list for easier evaluation:
        ps: List[MatrixExpr] = list(itertools.chain(*psChunked))
        allSymOrNums = len(ps) == len(list(filter(lambda e: isSymOrNum(e), ps)) )
        #allSyms = all(map(lambda expr : isSym(expr), ps))

        #if not allSymOrNums:
        outs = list(filter(lambda expr : not isSymOrNum(expr), outs))

        ins = list(filter(lambda expr : not isSymOrNum(expr), ins))
        # else just leave the syms as they are.


        # Zip the non-transp-out exprs with the transp-out expressions as list (NOTE: cannot be dictionary since we need to keep the same order of expressions as obtained from the preorder traversal, else substitution order will be messed up)
        outInPairs = list(zip(outs, ins))

        # Now must apply from the beginning to end, each of the expressions. Must replace in the first expr, all of the latter expressions, kind of like folding operation of Matryoshka dolls, to preserve all the end changes: C -> goes into -> B -> goes into -> A
        accFirst = expr #outs[0]

        f = lambda acc, outInPair: acc.xreplace({outInPair[1] : outInPair[0]})

        resultWithTypeOut = foldLeft(f , accFirst,  outInPairs) # outsNotsPairs[1:])

        return resultWithTypeOut



    if not expr.is_MatAdd: # TODO must change this to identify matmul or matsym exactly
        return algo_RippleOut_MatMul_or_MatSym(byType = byType, expr = expr)

    Constr: MatrixType = expr.func

    componentsOut: List[MatrixExpr] = list(map(lambda a: rippleOut(byType = byType, expr = a), expr.args))

    return Constr(*componentsOut)

# %%


# TODO this function cannot identify MatPow that is same as Inverse: for instance the obj D.I.T is MatPow of Transpose (not Inverse as expected) while using Transpose(Inverse(D)) is Transpose of Inverse, as expected.
# ---> So must find way for this function to recognize MatPow with NegativeOne and convert those into Inverse.
# ---> Need to find out if the double NegativeOne come from one set of MatPow inverse or not. Then replace accordingly with the Inverse constructor.
# TODO for now just avoid passing in D.I.T and just use the verbose constructor names.
def digger(expr: MatrixExpr) -> List[Tuple[List[MatrixType], MatrixExpr]]:
    '''Gets list of tuples, where each tuple contains expr list of types that surround the inner argument in the given matrix expression.

    EXAMPLE:

    Input: (((B*A*R)^-1)^T)^T

    Output: (ts, inner) where
        ts = [Transpose, Transpose, Inverse]
        inner = MatMul(B, A, R)
    '''

    #(csChunked, psChunked) = chunkExprsBy(byTypes = [Transpose ,Inverse], expr = expr)
    # Using a Constr_list we can add Trace, Derivative etc and any other constructor we wish.
    (csChunked, psChunked) = chunkExprsBy(byTypes = CONSTR_LIST, expr = expr)


    # Pair up the correct order of transpose types with the expressions
    # BEFORE: (Transpose in tsPs[0]) or (Inverse in tsPs[0])
    typeListExprListPair = list(filter(lambda tsPs : anyTypeIn(testTypes = CONSTR_LIST, searchTypes = tsPs[0]),
                                       list(zip(csChunked, psChunked))))

    # Get the first expression only, since it is the most layered, and pair its inner arg with the list of types from that pair.
    typeListInnerPair = list(map(lambda tsPs : (tsPs[0], inner(tsPs[1][0])), typeListExprListPair))

    return typeListInnerPair



# %% -------------------------------------------


# TEST 1:
expr = Transpose(Inverse(Inverse(MatMul(
    Inverse(Transpose(Inverse(R*C*D))),
    Inverse(Transpose(Inverse(Transpose(B*A*R))))
))))

res = digger(expr)
(resTypes, resExprs) = list(zip(*res)) # unzipping with zip (? why works?)
(resTypes, resExprs) = (list(resTypes), list(resExprs))


check = [
    ([Transpose, Inverse, Inverse], expr.arg.arg.args[0]),
    ([Inverse, Transpose, Inverse], R*C*D),
    ([Inverse, Transpose, Inverse, Transpose], B*A*R)
]
(checkTypes, checkExprs) = list(zip(*check))

showGroup([expr, resTypes, resExprs])
(checkTypes, checkExprs) = (list(checkTypes), list(checkExprs))

assert res == check
assert resExprs == checkExprs
# %%


def innerTrail(expr: MatrixExpr) -> List[Tuple[List[MatrixType], MatrixExpr]]:
    '''Gets the innermost expr (past all the .arg) on the first level only, and stores also the list of constructors'''

    def doInnerTrail(expr: MatrixExpr, accConstrs: List[MatrixType]) -> Tuple[List[MatrixType], MatrixExpr]:

        Constr = expr.func
        #types = [MatMul, MatAdd, MatrixSymbol, Symbol, Number]
        otherTypes = list( set(ALL_TYPES_LIST).symmetric_difference(CONSTR_LIST) )

        #isAnySubclass = any(map(lambda t : issubclass(Constr, t), otherTypes))

        #if (Constr in otherTypes) or isAnySubclass:
        if (Constr in otherTypes) or issubclass(Constr, Number):

            return (accConstrs, expr)

        # else keep recursing
        return doInnerTrail(expr = expr.arg, accConstrs = accConstrs + [Constr])

    return doInnerTrail(expr = expr, accConstrs = [])

# %%
# TEST 1
assert innerTrail(A) == ([], A)
# %%
# TEST 2
assert innerTrail(A.T) == ([Transpose], A)

assert innerTrail(Inverse(Transpose(Inverse(Inverse(A))))) == ([Inverse, Transpose, Inverse, Inverse], A)
# %%
# TEST 3
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
# TEST 4

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


# %%
def factor(expr: MatrixExpr, byType: MatrixType) -> MatrixExpr:

    typesInners = digger(expr)

    (types, innerExprs) = list(zip(*typesInners))
    (types, innerExprs) = (list(types), list(innerExprs))

    # Filtering the wrapper types that are `byType`s
    noSignalTypes: List[MatrixType] = list(map(lambda typeList:  list(filter(lambda t: t != byType, typeList)) , types))

    # Pair up all the types, filtered types, and inner expressions for easier querying later:
    triples: List[Tuple[List[MatrixType], List[MatrixType], List[MatrixExpr]]]  = list(zip(types, noSignalTypes, innerExprs))

    # Create new pairs from the filtered and inner Exprs, by attaching expr Transpose at the end if odd num else none.
    newTypesInners: List[Tuple[List[MatrixType], List[MatrixExpr]]] = list(map(lambda triple: ([byType] + triple[1],
                                                                                               triple[2]) if (triple[
                                                                                                                                                                                                                0].count(byType) % 2 == 1) else (triple[1], triple[2]) , triples))

    # Create the old exprs from the digger results:
    oldExprs: List[MatrixExpr] = list(map(lambda pair: applyTypesToExpr(pair), typesInners))

    # Create the new expressions with the simplified transposes:
    newExprs: List[MatrixExpr] = list(map(lambda pair: applyTypesToExpr(pair), newTypesInners))

    # Zip the old and new expressions for replacement later on:
    oldNewExprs: List[Dict[MatrixExpr, MatrixExpr]] = list(map(lambda pair: dict([pair]), list(zip(oldExprs, newExprs))))

    # Use fold to accumulate the results by replacing correct, simplified pieces of expressions into the overall expression.
    accFirst = expr
    f = lambda acc, oldNew: acc.xreplace(oldNew)
    result: MatrixExpr = foldLeft(f, accFirst, oldNewExprs)

    return result


# %% --------------------------------------------------------------


# NOTE: "single" means individual matrixsymbol
# NOTE: "grouped" means symbols gathered with an operation (+, -, *) and WRAPPED IN  constructors (trace, transpose, inverse ...)

def testCase_SLaGA_1a(algo, check: MatrixExpr, byType: MatrixType = None):

    # TEST 1a: SL + GA = single symbol + grouped MatAdd

    expr_SLaGA = MatAdd(A, Transpose(B + C.T) )
    # TODO how to assert that the algo has these arguments?
    params = [byType, expr_SLaGA] if not (byType == None) else [expr_SLaGA]

    res = algo(*params)
    #check = expr_SLaGA

    showGroup([
        expr_SLaGA,
        res,
        check
    ])

    assert expr_SLaGA.doit() == res.doit()
    assert res.doit() == check.doit() # even structurally equal
# %%
# TODO see the todo for matadd general addends error in the rippleout function
testCase_SLaGA_1a(rippleOut, MatAdd(A, Transpose(B + C.T) ), byType = Transpose)
# %%
# TODO have all the other functions here
# %% --------------------------------------------------------------

def testCase_SLaGA_1b(algo, check: MatrixExpr, byType = None):
    # TEST 1b: SL + GA = single symbol + grouped MatAdd (with more layerings per addend)

    expr_SLaGA = MatAdd(
        Inverse(Transpose(Transpose(A))),
        Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
            B , Inverse(Transpose(Transpose(E))) , C.T , R
        ))))) )
    )

    params = [byType, expr_SLaGA] if (not (byType == None)) else [expr_SLaGA]

    res = algo(*params)

    #res = algo(byType = byType, expr = expr_SLaGA)

    showGroup([
        expr_SLaGA,
        res,
        check
    ])

    # TODO why does expr.doit() come out with MatPow (1) instead of negative one for inverse?
    # assert expr_SLaGA.doit() == res.doit()
    assert equal(expr_SLaGA, res)
    assert res.doit() == check.doit()
# %%
check = MatAdd(
    Transpose(Transpose(Inverse(A))),
    Transpose(Transpose(Transpose(Inverse(Inverse(MatAdd(
        Transpose(Transpose(Inverse(E))), B, R, C.T
    ))))))
)
testCase_SLaGA_1b(rippleOut, check, byType = Transpose)
# %% --------------------------------------------------------------


def testCase_SLmGA_2a(algo, check: MatrixExpr, byType = None):
    # TEST 2a: SL * GA = single symbol * grouped MatAdd

    expr_SLmGA = MatMul(
        A,
        Inverse(Transpose(MatAdd(B, C.T)))
    )

    #res = algo(byType = byType , expr = expr_SLmGA)
    params = [byType, expr_SLmGA] if (not (byType == None)) else [expr_SLmGA]

    res = algo(*params)

    showGroup([
        expr_SLmGA,
        res,
        check
    ])

    #TODO matpow error again
    # assert expr_SLmGA.doit() == res.doit()
    assert equal(expr_SLmGA, res)
    assert res.doit() == check.doit()
# %%
check = MatMul(
    A,
    Transpose(Inverse(MatAdd(B, C.T)))
)
testCase_SLmGA_2a(rippleOut, check, byType = Transpose)
# %%
check = MatMul(
    A,
    Transpose(Inverse(MatAdd(B, C.T)))
)
testCase_SLmGA_2a(factor, check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 2b: SL * GA = single symbol * grouped MatAdd (with more layerings per addend)

expr_SLmGA = MatMul(
    Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) )
)

res = factor(expr_SLmGA)

check = MatMul(
    Inverse(Inverse(Inverse(A))),
    Transpose(Inverse(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    ))))
)

showGroup([
    expr_SLmGA,
    res,
    check
])

# TODO matpow error again
# assert expr_SLmGA.doit() == res.doit()
# TODO doesn't work
# assert equal(expr_SLmGA, res)
assert res.doit() == check.doit()
# %% --------------------------------------------------------------


# TEST 3a: SL + GM = single symbol + grouped MatMul

expr_SLaGM = MatAdd(A, MatMul(B, A.T, R.I))
res = factor(expr_SLaGM)
check = expr_SLaGM

showGroup([
    expr_SLaGM,
    res,
    check
])

# TODO matpow error
assert expr_SLaGM.doit() == res.doit()
#assert equal(expr_SLaGM, res)
assert res.doit() == check.doit()
# %% --------------------------------------------------------------


# TEST 3b: SL + GM = single symbol + grouped MatMul (with more layerings per addend)

expr_SLaGM = MatAdd(
    Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) )
)

res = factor(expr_SLaGM)

check = MatAdd(
    Inverse(Inverse(Inverse(A))),
    Transpose(Inverse(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    ))))
)

showGroup([
    expr_SLaGM,
    res,
    check
])

# TODO matpow error?
# assert expr_SLaGM.doit() == res.doit()
assert equal(expr_SLaGM, res)
assert res.doit() == check.doit()

# %% --------------------------------------------------------------


# TEST 4a: SL * GM = single symbol * grouped MatMul

expr_SLmGM = MatMul(A, MatMul(B.T, A, R.I))
res = factor(expr_SLmGM)
check = expr_SLmGM

showGroup([
    expr_SLmGM,
    res,
    check
])

assert expr_SLmGM.doit() == res.doit()
assert res.doit() == check.doit()
# %% --------------------------------------------------------------


# TEST 4b: SL * GM = single symbol * grouped MatMul (with more layerings per multiplicand)

expr_SLmGM = MatMul(
    Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatMul(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) )
)

res = factor(expr_SLmGM)

check = MatMul(
    Inverse(Inverse(Inverse(A))),
    Transpose(Inverse(Inverse(MatMul(
        B, E.I, C.T, Transpose(Inverse(R))
    ))))
)

showGroup([
    expr_SLmGM,
    res,
    check
])

assert expr_SLmGM.doit() == res.doit()
assert res.doit() == check.doit()

# %% --------------------------------------------------------------


# TEST 5a: SA + GA = single symbol Add + grouped Matadd

expr_SAaGA = MatAdd(
    A.T, B.I, C,
    Inverse(Transpose(Inverse(MatAdd(
        B, C.T, D.I
    ))))
)
res = factor(expr_SAaGA)
check = MatAdd(
    A.T, B.I, C,
    Transpose(Inverse(Inverse(MatAdd(
        B, C.T, D.I
    ))))
)

showGroup([
    expr_SAaGA,
    res,
    check
])

# TODO matpow error again
# assert expr_SAaGA.doit() == res.doit()
assert equal(expr_SAaGA, res)
assert res.doit() == check.doit()
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

res = factor(expr_SAaGA)

check = MatAdd(
    Inverse(Inverse(Inverse(A))),
    Inverse(Inverse(B)),
    C,
    Transpose(Inverse(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    ))))
)

showGroup([
    expr_SAaGA,
    res,
    check
])

# TODO matpow error
# assert expr_SAaGA.doit() == res.doit()
assert equal(expr_SAaGA, res)
assert res.doit() == check.doit()


# %% --------------------------------------------------------------


# TEST 6a: SA * GA = single symbol Add * grouped Matadd

expr_SAmGA = MatMul(
    A.T, B.I, C,
    Inverse(Transpose(Inverse(MatAdd(
        B, C.T, D.I
    ))))
)

res = factor(expr_SAmGA)

check = MatMul(
    A.T, B.I, C,
    Transpose(Inverse(Inverse(MatAdd(
        B, C.T, D.I
    ))))
)

showGroup([
    expr_SAmGA,
    res,
    check
])

# TODO matpow error
# assert expr_SAmGA.doit() == res.doit()
# TODO why doesn't this work?
# assert equal(expr_SAmGA, res)
assert res.doit() == check.doit()
# TODO why does check.doit() not show inverse being distributed for all terms D, C, B, just for D? URGENT
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

res = factor(expr_SAmGA)

check = MatMul(
    Inverse(Inverse(Inverse(A))),
    Inverse(Inverse(B)),
    C,
    Transpose(Inverse(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    ))))
)

showGroup([
    expr_SAmGA,
    res,
    check
])

# TODO matpow error?
# assert expr_SAmGA.doit() == res.doit()
# TODO why doesn't this work?
# assert equal(expr_SAmGA, res)
assert res.doit() == check.doit()


# %% --------------------------------------------------------------


# TEST 7a: SA + SM = single symbol Add + single symbol Mul

expr_SAaSM = MatAdd(
    A.T, B.I, C,
    Inverse(Transpose(Inverse(MatMul(
        B, C.T, D.I, E
    ))))
)
res = factor(expr_SAaSM)
check = MatAdd(
    A.T, B.I, C,
    Transpose(Inverse(Inverse(MatMul(
        B, C.T, D.I, E
    ))))
)

showGroup([
    expr_SAaSM,
    res,
    check
])

assert expr_SAaSM.doit() == res.doit()
assert res.doit() == check.doit()
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

res = factor(expr_SAaSM)

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

showGroup([
    expr_SAaSM,
    res,
    check
])

assert expr_SAaSM.doit() == res.doit()
assert res.doit() == check.doit()


# %% --------------------------------------------------------------


# TEST 8a: SM + GA = single symbol Mul + group symbol Add

expr_SMaGA = MatAdd(
    MatMul(A.T, B.I, C, D.T),
    Inverse(Transpose(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

res = factor(expr_SMaGA)

check = MatAdd(
    MatMul(A.T, B.I, C, D.T),
    Transpose(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

showGroup([
    expr_SMaGA,
    res,
    check
])

# TODO matpow error
# assert expr_SMaGA.doit() == res.doit()
assert equal(expr_SMaGA, res)
assert res.doit() == check.doit()
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

res = factor(expr_SMaGA)

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

showGroup([
    expr_SMaGA,
    res,
    check
])

# TODO matpow error
# assert expr_SMaGA.doit() == res.doit()
assert equal(expr_SMaGA, res)
assert res.doit() == check.doit()


# %% --------------------------------------------------------------


# TEST 9a: SM * GA = single symbol Mul * group symbol Add

expr_SMmGA = MatMul(
    MatMul(A.T, B.I, C, D.T),
    Inverse(Transpose(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

res = factor(expr_SMmGA)

check = MatMul(
    MatMul(A.T, B.I, C, D.T),
    Transpose(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

showGroup([
    expr_SMmGA,
    res,
    check
])

# TODO matpow error
# assert expr_SMmGA.doit() == res.doit()
assert equal(expr_SMmGA, res)
assert res.doit() == check.doit()
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

res = factor(expr_SMmGA)

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

showGroup([
    expr_SMmGA,
    res,
    check
])

# TODO matpow error
# assert expr_SMmGA.doit() == res.doit()
# TODO this doesn't work
# assert equal(expr_SMmGA, res)
assert res.doit() == check.doit()



# %% --------------------------------------------------------------


# TEST 10a: SM + GM = single symbol Mul + group symbol Mul

expr_SMaGM = MatAdd(
    MatMul(A.T, B.I, C, D.T),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), C.T
    )))
)

res = factor(expr_SMaGM)

check = MatAdd(
    MatMul(A.T, B.I, C, D.T),
    Transpose(Inverse(MatMul(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

showGroup([
    expr_SMaGM,
    res,
    check
])

assert expr_SMaGM.doit() == res.doit()
assert res.doit() == check.doit()
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

res = factor(expr_SMaGM)

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

showGroup([
    expr_SMaGM,
    res,
    check
])


assert expr_SMaGM.doit() == res.doit()
assert res.doit() == check.doit()



# %% --------------------------------------------------------------


# TEST 11a: SM * GM = single symbol Mul * group symbol Mul

expr_SMmGM = MatMul(
    MatMul(A.T, B.I, C, D.T),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), C.T
    )))
)

res = factor(expr_SMmGM)

check = MatMul(
    MatMul(A.T, B.I, C, D.T),
    Transpose(Inverse(MatMul(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

showGroup([
    expr_SMmGM,
    res,
    check
])

assert expr_SMmGM.doit() == res.doit()
assert res.doit() == check.doit()
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

res = factor(expr_SMmGM)

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

showGroup([
    expr_SMmGM,
    res,
    check
])

assert expr_SMmGM.doit() == res.doit()
assert res.doit() == check.doit()


# %% --------------------------------------------------------------


# TEST 12a: GA + GM = group symbol Add + group symbol Mul

expr_GAaGM = MatAdd(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), C.T
    )))
)

res = factor(expr_GAaGM)

check = MatAdd(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Transpose(Inverse(MatMul(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

showGroup([
    expr_GAaGM,
    res,
    check
])

assert expr_GAaGM.doit() == res.doit()
assert res.doit() == check.doit()
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

res = factor(expr_GAaGM)

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

showGroup([
    expr_GAaGM,
    res,
    check
])

assert expr_GAaGM.doit() == res.doit()
assert res.doit() == check.doit()


# %% --------------------------------------------------------------


# TEST 13a: GA * GM = group symbol Add * group symbol Mul

expr_GAmGM = MatMul(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), C.T
    )))
)

res = factor(expr_GAmGM)

check = MatMul(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Transpose(Inverse(MatMul(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

showGroup([
    expr_GAmGM,
    res,
    check
])

assert expr_GAmGM.doit() == res.doit()
assert res.doit() == check.doit()
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

res = factor(expr_GAmGM)

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

showGroup([
    expr_GAmGM,
    res,
    check
])

assert expr_GAmGM.doit() == res.doit()
assert res.doit() == check.doit()


# %% --------------------------------------------------------------


# TEST 14a: GM + GM = group symbol Mul + group symbol Mul

expr_GMaGM = MatAdd(
    Inverse(MatMul(A.T, B.I, C, D.T)),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), C.T
    )))
)

res = factor(expr_GMaGM)

check = MatAdd(
    Inverse(MatMul(A.T, B.I, C, D.T)),
    Transpose(Inverse(MatMul(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

showGroup([
    expr_GMaGM,
    res,
    check
])

assert expr_GMaGM.doit() == res.doit()
assert res.doit() == check.doit()
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

res = factor(expr_GMaGM)

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

showGroup([
    expr_GMaGM,
    res,
    check
])

assert expr_GMaGM.doit() == res.doit()
assert res.doit() == check.doit()


# %% --------------------------------------------------------------


# TEST 15a: GM * GM = group symbol Mul * group symbol Mul

expr_GMmGM = MatMul(
    Inverse(MatMul(A.T, B.I, C, D.T)),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), C.T
    )))
)

res = factor(expr_GMmGM)

check = MatMul(
    Inverse(MatMul(A.T, B.I, C, D.T)),
    Transpose(Inverse(MatMul(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

showGroup([
    expr_GMmGM,
    res,
    check
])

assert expr_GMmGM.doit() == res.doit()
assert res.doit() == check.doit()


# %% --------------------------------------------------------------


# TEST 15b: GM * GM = group symbol Mul * group symbol Mul

expr_GMmGM = MatMul(
    Transpose(MatMul(A, B)),
    Transpose(MatMul(R, J))
)

res = factor(expr_GMmGM)

check = expr_GMmGM

showGroup([
    expr_GMmGM,
    res,
    check
])

assert expr_GMmGM.doit() == res.doit()
assert res.doit() == check.doit()




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

res = factor(expr_GMmGM)

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

showGroup([
    expr_GMmGM,
    res,
    check
])

assert expr_GMmGM.doit() == res.doit()
assert res.doit() == check.doit()


# %%

#TODO later: SL_a_GA_2 as innermost arg layered with transpse and then combine with any other kind: SL, SA, SM, GA, GM in between the layers. TODO for every kind of expression

# TODO 2: must take all above tests and organize so each function I made gets all those above cases (then separate the "special" cases per function beneath these "general" cases which are made just by combining types of expressions, as opposed to the "biased"/"special" cases.)

# %%



def algoTransposeWrap_shallow(innerExpr: MatrixExpr) -> MatrixExpr:
    '''The wrapping algo for taking expr set of simple arguments and enveloping them in transpose.'''

    Constr: MatrixType = innerExpr.func
    #assert Constr in [MatAdd, MatMul]

    numArgsTranspose: int = len(list(filter(lambda a: a.is_Transpose, innerExpr.args )))

    # Building conditions for checking if we need to wrap the expr, or else return it as is.
    mostSymsAreTranspose: bool = (numArgsTranspose / len(innerExpr.args) ) >= 0.5

    # NOTE: len == 0 of free syms when Number else for MatSym len == 1 so put 0 case just for safety.
    #onlySymComponents_AddMul = lambda expr: (expr.func in [MatAdd, MatMul]) and all(map(lambda expr: len(expr.free_symbols) in [0, 1], expr.args))

    mustWrap: bool = (Constr in [MatAdd, MatMul]) and mostSymsAreTranspose #and onlySymComponents_AddMul(innerExpr)

    if not mustWrap:
        return innerExpr

    # Else do the wrapping algorithm:
    invertedArgs: List[MatrixExpr] = list(map(lambda a: pickOut(a), innerExpr.args))


    invertedArgs: List[MatrixExpr] = list(reversed(invertedArgs)) if Constr == MatMul else invertedArgs

    wrapped: MatrixExpr = Transpose(Constr(*invertedArgs))

    return wrapped

# %%


isSym = lambda m: len(m.free_symbols) in [0,1]

# NOTE: len == 0 of free syms when Number else for MatSym len == 1 so put 0 case just for safety.
#onlySymComponents_AddMul = lambda expr: (expr.func in [MatAdd, MatMul]) and all(map(lambda expr: len(expr.free_symbols) in [0, 1], expr.args))

isSimpleArgs = lambda e: all(map(lambda a: len(a.free_symbols) in [0, 1], e.args))
isInnerExpr = lambda e: (e.func in [MatAdd, MatMul]) and isSimpleArgs(e)

def algoTransposeWrap_deep(expr: MatrixExpr) -> MatrixExpr:

    Constr = expr.func

    if isSym(expr):
        return expr
    elif Constr in [MatAdd, MatMul]:  #then split the polarizing operation over the arguments since any one of the args can be an inner expr.
        wrappedArgs: List[MatrixExpr] = list(map(lambda a: algoTransposeWrap_deep(a), expr.args))
        exprWithPartsWrapped: MatrixExpr = Constr(*wrappedArgs)
        exprOverallWrapped: MatrixExpr = algoTransposeWrap_shallow(exprWithPartsWrapped)

        return exprOverallWrapped

    elif isInnerExpr(expr):
        wrappedExpr: MatrixExpr = algoTransposeWrap_shallow(expr)
        return wrappedExpr
    else: # else is Trace, Transpose, or Inverse or any other constructor
        innerExpr = expr.arg

        return Constr( algoTransposeWrap_deep(innerExpr) )
# %%
e = Inverse(MatMul(
    Transpose(Inverse(Transpose(MatAdd(B.T, A.T, R, MatMul(Transpose(Inverse(B*A*R.T)), MatAdd(E, J, D)), Inverse(Transpose(E)), Inverse(Transpose(D)))))),
    Inverse(Transpose(MatMul(A.T, B.T, E.I, Transpose(Inverse(Transpose(A + E + R.T))), C)))
))
e1 = e.arg.args[1].arg.arg

#algoTransposeWrap_deep(e1)
# algoTransposeWrap_deep(e) # doesn't factor out the transposes not bring them outer

re = rippleOut(e)
fre = factor(re)
wfre = algoTransposeWrap_deep(fre)

assert wfre.doit() == e.doit()

showGroup([
    e, re, fre, wfre
])
# %%



# Making expr group transpose that goes deep until innermost expression (largest depth) and applies the group algo there
# Drags out even the inner transposes (aggressive grouper)
def polarizeTranspose(expr: MatrixExpr) -> MatrixExpr:
    '''Given an expression with innermost args nested as components inside another expression, (many nestings), this function tries to drag / pull / force out all the transposes from the groups of expressions of matmuls / matmadds and from individual symbols.

    Tries to create one nesting level that mentions transpose (there can be other nestings with inverse inside for instance but the outermost nesting must be the only one with transpose).

    There must be no layering of transpose in the nested expressions in the result returned by this function -- polarization of transpose to the outer edges.'''


    # Need to factor out transposes and ripple them out first before passing to wrap algo because the wrap algo won't reach in and factor or bring out inner transposes, will just overlay on top of them without simplifying (bad, since yields more complicated expression)
    fe = factor(expr) # no need for ripple out because factor does this implicitly

    wfe = algoTransposeWrap_deep(fe)

    # Must bring out the extra transposes that are brought out by the wrap algo and then cut out extra ones.
    fwfe = factor(wfe)

    return fwfe


# %% -------------------------------------------------------------




# TEST 1: inverse out, transpose in
expr = Inverse(Transpose(C*E*B))

# %%
res = group(expr)
check = expr

showGroup([
    expr, res, check
])

assert equal(res, check)
assert equal(expr.doit(), res.doit())
# %%


res = rippleOut(expr)
check = Transpose(Inverse(MatMul(C, E, B)))

showGroup([
    expr, res, check
])

assert equal(res, check)
assert equal(expr.doit(), res.doit())
# %% -------------------------------------------------------------


# TEST 2: transpose out, inverse in
expr = Transpose(Inverse(C*E*B))

# %%
res = group(expr)
check = expr

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %%
res = rippleOut(expr)
check = Transpose(Inverse(C*E*B))

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %% -------------------------------------------------------------


# TEST 3: individual transposes inside inverse
expr = Inverse(B.T * E.T * C.T)
# %%

res = group(expr)
check = Inverse(Transpose(C*E*B))

showGroup([
    expr, res, check
])
assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %%

res = rippleOut(expr)
check = expr

#checkAggr = Transpose(Inverse(C*E*B))

showGroup([
    expr, res, check
])
assert equal(expr.doit(), res.doit())
assert equal(check, res)
# %% -------------------------------------------------------------


# TEST 4: individual inverse inside transpose

expr = Transpose(B.I * E.I * C.I)
# %%

res = group(expr)
check = expr

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)

# %%

res = rippleOut(expr)
check = expr

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %% -------------------------------------------------------------


# TEST 5 expr: individual symbols

Q = MatrixSymbol("Q", a, b)

# %%

(expr1, check1) = (Q, Q)
(expr2, check2) = (Q.T, Q.T)
(expr3, check3) = (C.I, C.I)
(expr4, check4) = (Inverse(Transpose(C)), Inverse(Transpose(C)))

res1 = group(expr1)
res2 = group(expr2)
res3 = group(expr3)
res4 = group(expr4)

showGroup([
    (expr1, res1, check1),
    (expr2, res2, check2),
    (expr3, res3, check3),
    (expr4, res4, check4)
])

assert equal(res1, check1)
assert equal(res2, check2)
assert equal(res3, check3)
assert equal(res4, check4)

# %%


(expr1, check1) = (Q, Q)
(expr2, check2) = (Q.T, Q.T)
(expr3, check3) = (C.I, C.I)
(expr4, check4) = (Inverse(Transpose(C)), Transpose(Inverse(C)))

res1 = rippleOut(expr1)
res2 = rippleOut(expr2)
res3 = rippleOut(expr3)
res4 = rippleOut(expr4)

showGroup([
    (expr1, res1, check1),
    (expr2, res2, check2),
    (expr3, res3, check3),
    (expr4, res4, check4)
])

assert equal(res1, check1)
assert equal(res2, check2)
assert equal(res3, check3)
assert equal(res4, check4)



# %% -------------------------------------------------------------



# TEST 5 b: inidivudal symbols nested

expr = Transpose(Inverse(Inverse(Inverse(Transpose(MatMul(
    A,
    Inverse(Transpose(Inverse(Transpose(C)))),
    Inverse(Transpose(R)),
    M
))))))
# %%

res = group(expr)


check = Transpose(Inverse(Inverse(Inverse(Transpose(Transpose(MatMul(
    M.T,
    Transpose(Inverse(Transpose(R))),
    Transpose(Inverse(Transpose(Inverse(Transpose(C))))),
    A.T
)))))))

showGroup([
    expr,
    res,
    check
])

assert res.doit() == expr.doit()
assert equal(res, check)

# %%


res = rippleOut(expr)

check = Transpose(Transpose(Inverse(Inverse(Inverse(MatMul(
    A,
    Transpose(Transpose(Inverse(Inverse(C)))),
    Transpose(Inverse(R)),
    M
))))))

showGroup([
    expr,
    res,
    check
])

assert res.doit() == expr.doit()
assert equal(res, check)


# %% -------------------------------------------------------------




# TEST 6: grouped products

expr = MatMul( Transpose(A*B), Transpose(R*J) )
# %%

res = group(expr)
check = Transpose(R*J*A*B)

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
# TODO if you want you should split the matmul case into expr "simple matmul case" in that there are no innermost nestings and everything is on the first level.
assert equal(res, check)

# %%

res = rippleOut(expr)
check = expr

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %% -------------------------------------------------------------


# TEST 7: individual transposes littered along as matmul

expr = B.T * A.T * J.T * R.T

# %%

res = group(expr)
check = Transpose(R*J*A*B)

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %%

res = rippleOut(expr)
check = expr

showGroup([expr, res, check])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %% -------------------------------------------------------------


# TEST 8: inverses mixed with transpose in expr matmul, but with transposes all as the outer expression

expr = MatMul(A , Transpose(Inverse(R)), Transpose(Inverse(L)) , K , E.T , B.I )
# %%

res = group(expr)

check = Transpose(
    MatMul( Transpose(Inverse(B)), E , K.T , Inverse(L) , Inverse(R) , A.T)
)

showGroup([
    expr, res, check ,
    res.doit(),
    check.doit()
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)

# %%

res = rippleOut(expr)

check = expr

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)


# %% -------------------------------------------------------------


# TEST 9: mix of inverses and transposes in expr matmul, but this time with transpose not as outer operation, for at least one symbol case.

expr = MatMul(A , Transpose(Inverse(R)), Inverse(Transpose(L)) , K , E.T , B.I )
# %%

res = group(expr)

check = Transpose(
    MatMul( Transpose(Inverse(B)), E , K.T , Transpose(Inverse(Transpose(L))) , R.I , A.T)
)

showGroup([
    expr, res, check ,
    res.doit(),
    check.doit()
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)

# %%

res = rippleOut(expr)

check = MatMul(
    A, Transpose(Inverse(R)), Transpose(Inverse(L)), K, E.T, B.I
)

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %% -------------------------------------------------------------


# TEST 10: transposes in matmuls and singular matrix symbols, all in expr matadd expression.

expr = A * R.T * L.T * K * E.T * B + D.T + K
# %%

res = group(expr)
resGroup = group(expr, combineAdds = True)

check = MatAdd( Transpose(MatMul(B.T, E, K.T, L, R, A.T)) , D.T, K)
checkGroup = Transpose( B.T * E * K.T * L * R * A.T + D + K.T)

showGroup([
    (expr, res, check),
    (expr, resGroup, checkGroup)
])

assert expr.doit() == res.doit()
assert expr.doit() == resGroup.doit()
assert equal(res, check)
assert equal(resGroup, checkGroup)
# %%

res = rippleOut(expr)
check = expr

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %% -------------------------------------------------------------


# TEST 11: digger case, very layered expression (with transposes separated so not expecting the grouptranspose to change them). Has inner arg matmul.

expr = Trace(Transpose(Inverse(Transpose(C*D*E))))
# %%

res = group(expr)
check = expr

showGroup([
    expr, res, check
])

# TODO fix the expandMatMul function so that trace can be passed as argument (just do expand on what is inside)
#assert equal(res, check)
#assert equal(expr.doit(), res.doit())

assert res == check
# %%

res = rippleOut(expr)
check = Trace(Transpose(Transpose(Inverse(C*D*E))))

showGroup([
    expr, res, check
])

# TODO make the expandMatMul function work with Trace (throws error here because Trace has no 'shape' attribute)
# assert equal(expr.doit(), res.doit())
assert expr.doit() == res.doit()
# TODO assert res == check # make structural equality work
#assert equal(res, check)
# %% -------------------------------------------------------------



# TEST 12: very layered expression, but transposes are next to each other, with inner arg as matmul

expr = Trace(Transpose(Transpose(Inverse(C*D*E))))

res1 = group(expr)

res2 = rippleOut(expr)

check = expr

showGroup([
    expr, res1, res2, check
])

assert expr.doit() == res1.doit()
assert expr.doit() == res2.doit()
# assert equal(res, check) # todo fix equals function
assert res1 == check
assert res2 == check

# %% -------------------------------------------------------------


# TEST 13: very layered expression (digger case) with individual transpose and inverses littered in the inner matmul arg.
expr = Transpose(Inverse(Transpose(C.T * A.I * D.T)))
# %%


res = group(expr)

check = Transpose(Inverse(Transpose(Transpose(
    MatMul(D , Transpose(Inverse(A)), C )
))))

showGroup([
    expr, res, check ,
    res.doit(),
    check.doit()
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %%

res = rippleOut(expr)

check = Transpose(Transpose(Inverse(
    C.T * A.I * D.T
)))

showGroup([
    expr, res, check
])

assert expr.doit() == res.doit()
assert equal(res, check)
# %% -------------------------------------------------------------



# TEST 14 expr: Layered expression (many innermost nestings). The BAR expression has transpose outer and inverse inner, while the other transposes are outer already. (ITIT)

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

res = group(expr)

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

showGroup([expr, res, check])

assert equal(res.doit(), expr.doit())
assert equal(res, check)

# %%


res = rippleOut(expr)
check = expr

showGroup([
    expr, res, check
])

assert expr.doit() == res.doit()
assert equal(res, check)



# %%

# TODO update the grouptranspose reverse algo to deal with minimum number of transpose elimination scheme (of innermosts across the same level (?)). (Use digger?)
# TODO: must put this reverse algo directly inside the transpose out function, to adapt the result and avoid having product transposes, to aim for having just one innermost expression with transposes surrounding.

#res = transposeOut(expr)

# Aggressive check
check = Transpose(Transpose(Transpose(Inverse(MatMul(
    A, D.T, R.T,
    Inverse(MatMul(B, Inverse(B*A*R), Transpose(Inverse(E)), C))
)))))
check
# %% -------------------------------------------------------------




# TEST 14 b: layered expression (many innermost nestings). The BAR expression has transpose inner and inverse outer and other transposes are outer already, after inverse. (ITIT)

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

res = group(expr)

check = Transpose(Inverse(Transpose(MatMul(
    Inverse(Transpose(MatMul(
        B, Transpose(Inverse(Transpose(B*A*R))), Transpose(Inverse(E)), C
    ))),
    Transpose(Transpose(MatMul(R, D, A.T)))
))))

showGroup([expr, res, check])

assert expr.doit() == res.doit()
assert equal(res, check)



# %%


res = rippleOut(expr)

# Simple check
check = Transpose(Inverse(
    MatMul(A*D.T*R.T ,
        Transpose(
            Inverse(
                MatMul(C.T, E.I, Transpose(Inverse(B*A*R)), B.T)
            )
        )
    )
))
# Aggressive check
#check = Transpose(Transpose(Transpose(Inverse(MatMul(
#    A, D.T, R.T,
#    Inverse(MatMul(B, Inverse(B*A*R), Transpose(Inverse(E)), C))
#)))))

showGroup([
    expr, res, check
])

assert expr.doit() == res.doit()
assert equal(res, check)



# %% -------------------------------------------------------------


# TEST 14 c: innermost layered expression, with BAR having tnraspose outer and inverse inner, and the other expressions have transpose inner and inverse outer (TITI)
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

res = group(expr)

# TODO how to make this grunction avoid getting this unnecessary transpose?
# Desired check:
#checkDesired = Inverse(Transpose(MatMul(
#    MatMul(A, D.T, R.T,
#        Inverse(Transpose(Transpose(
#            B, Inverse(B*A*R), Transpose(Inverse(E)), C
#        )))
#    )
#)))
check = Inverse(Transpose(Transpose(MatMul(
    Transpose(Inverse(Transpose(Transpose(MatMul(
        B, Inverse(B*A*R), Transpose(Inverse(E)), C
    ))))),
    Transpose(Transpose(MatMul(R, D, A.T)))
))))

showGroup([expr, res, check])

#assert equal(res.doit(), expr.doit())
assert res.doit() == expr.doit()
assert equal(res, check)



# %%


res = rippleOut(expr)

# Aggressive check
#check = Transpose(Transpose(Transpose(Inverse(MatMul(
#    A, D.T, R.T,
#    Inverse(MatMul(B, Inverse(B*A*R), Transpose(Inverse(E)), C))
#)))))
# Simple check
check = Transpose(Inverse(MatMul(
    A, D.T, R.T,
    Transpose(Inverse(MatMul(
        C.T, E.I, Transpose(Inverse(B*A*R)), B.T
    )))
)))

showGroup([
    expr, res, check
])

assert expr.doit() == res.doit()
assert equal(res, check)



# %% -------------------------------------------------------------


# TEST 14 d: innermost layered expression, with BAR having tnraspose inner and inverse outer, and the other expressions have transpose inner and inverse outer (TITI)

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

res = group(expr)

# TODO how to make this grunction avoid getting this unnecessary transpose?

check = Inverse(Transpose(Transpose(MatMul(
    Transpose(Inverse(Transpose(Transpose(MatMul(
        B, Transpose(Inverse(Transpose(B*A*R))), Transpose(Inverse(E)), C
    ))))),
    Transpose(Transpose(MatMul(R, D, A.T)))
))))

showGroup([expr, res, check])

#assert equal(res.doit(), expr.doit())
assert res.doit() == expr.doit()
assert equal(res, check)


# %%


res = rippleOut(expr)

# Simple check
check = Transpose(Inverse(MatMul(
    A, D.T, R.T,
    Transpose(Inverse(MatMul(
        C.T, E.I, Transpose(Inverse(B*A*R)), B.T
    )))
)))

# Aggressive check
#check = Transpose(Transpose(Transpose(Inverse(MatMul(
#    A, D.T, R.T,
#    Inverse(MatMul(B, Inverse(B*A*R), Transpose(Inverse(E)), C))
#)))))

showGroup([
    expr, res, check
])

assert expr.doit() == res.doit()
assert equal(res, check)



# %% -------------------------------------------------------------



# TEST 15: very layered expression, and the inner arg is matadd with some matmuls but some matmuls have one of the elements as another layered arg (any of the test 14 cases, like 14 expr), so we can test if the function reaches all rabbit holes effectively.

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

res = group(expr)

# TODO need to update this test since I want the group transpose to call transpose out to make sure the transpose gets simplified, and not just re-applied  doubly
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

showGroup([
    expr,
    res,
    check,
    expr.doit(),
    res.doit(),
    check.doit()
])

assert equal(expr, res)
#assert check == res # TODO why not? they should equal structurally
# # TODO convert this test to the structural equal test to make sure the structure is the same
assert equal(res, check)



# %%


res = rippleOut(expr)

# Simple check 14 expr
check_inner_14a = Inverse(
    MatMul(A*D.T*R.T ,
        Transpose(
            Inverse(
                MatMul(C.T, E.I, L, B.T)
            )
        )
    )
)

# Aggressive check 14 expr
#check_inner_14a = Transpose(Transpose(Inverse(MatMul(
#    A, D.T, R.T,
#    Inverse(MatMul(B, Inverse(B*A*R), Transpose(Inverse(E)), C))
#))))

check = Transpose(MatAdd(
    B.T * A.T * J.T * R.T,
    check_inner_14a
))

showGroup([
    expr, res, check
])

assert expr.doit() == res.doit()
assert equal(res, check)




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

