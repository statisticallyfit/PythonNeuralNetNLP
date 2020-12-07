
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
C = MatrixSymbol('C', c, c)
E = MatrixSymbol('E', c, c)
B = MatrixSymbol('B', c, c)
L = MatrixSymbol('L', c, c)
D = MatrixSymbol('D', c, c)

# Testing
expr = C*E*B*E*L*E*D 

assert splitOnce(expr.args, E, 1) == ([C], [B, E, L, E, D])    
assert splitOnce(expr.args, E, 2) == ([C, E, B], [L, E, D])
assert splitOnce(expr.args, E, 3) == ([C, E, B, E, L], [D])
assert splitOnce(expr.args, E, 0) == ([], [])
assert splitOnce(expr.args, E, 4) == ([], [])
# TODO how to assert error for negative number n?





# %%
hasTranspose = lambda e : "Transpose" in srepr(e)

def pickOut(a):
    if not hasTranspose(a):
        #return Transpose(MatMul(*a.args))
        return Transpose(a)
    elif hasTranspose(a) and a.is_Transpose:
        return a.arg 
    return Transpose(a) #TODO check 

def groupTranpose_MatMul_or_MatSym(expr: MatrixExpr) -> MatrixExpr:

    def revTransposeAlgo(expr):
        '''Converts B.T * A.T --> (A * B).T'''

        if hasTranspose(expr) and expr.is_MatMul:

            revs = list(map(lambda a : pickOut(a), reversed(expr.args)))

            return Transpose(MatMul(*revs))
            
        return expr 

    ps = list(preorder_traversal(expr))
    ms = list(filter(lambda p : p.is_MatMul, ps))
    ts = list(map(lambda m : revTransposeAlgo(m), ms))

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
C = MatrixSymbol('C', c, c)
D = MatrixSymbol('D', c, c)
E = MatrixSymbol('E', c, c)
A = MatrixSymbol('A', c, c)
B = MatrixSymbol('B', c, c)
R = MatrixSymbol('R', c, c)

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

res = groupTranpose_MatMul_or_MatSym(expr)

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


def groupTranspose(expr: MatrixExpr, combineAdds:bool = False) -> MatrixExpr: 
    '''Combines transposes when they are the outermost operations. 
    
    Brings the individual transpose operations out over the entire group (factors out the transpose and puts it as the outer operation). 
    NOTE: This happens only when the transpose ops are at the same level. If they are nested, that "bringing out" task is left to the transposeOut function. 
    
    combineAdds:bool  = if True then the function will group all the addition components under a transpose, like in matmul, otherwise it will group only the individual components of the MatAdd under transpose.'''

    #isMatSym = lambda s: len(s.free_symbols) == 1

    if not expr.is_MatAdd: 
        return groupTranpose_MatMul_or_MatSym(expr)

    addendsTransp: List[MatrixExpr] = list(map(lambda a: groupTranspose(a), expr.args))

    if combineAdds:
        innerAddends = list(map(lambda t: pickOut(t), addendsTransp))
        return Transpose(MatAdd(*innerAddends))
    # Else not combining adds, just grouping transposes in addends individually
    return MatAdd(*addendsTransp)

    #elif expr.is_MatMul or isMatSym(expr):
    #    return groupTranspose(expr)

    #else: # is transpose or inverse or trace etc and need to dig to get the innermost argument
    
    
    

# %%




### TRANSPOSE OUT HERE -----------------------------------------
from sympy.core.assumptions import ManagedProperties

def chunkInvTrans(constructors: List[ManagedProperties]) -> List[List[ManagedProperties]]:
    '''Separates the Inverse and Transpose types in a list of types, and keeps other types separate too, just as how they appear in the original list'''

    # Not strictly necessary here, just useful if you want to see shorter version of names below
    #getSimpleTypeName = lambda t : str(t).split("'")[1].split(".")[-1]

    # Step 1: coding up in pairs for easy identification: need Inverse and Transpose tagged as same kind
    codeConstrPairs: List[(int, ManagedProperties)] = list(map(lambda c : (0, c) if c == Transpose or c == Inverse else (1, c), constructors))

    # Step 2: getting the groups
    chunkedPairs: List[List[(int, ManagedProperties)]] = [list(group) for key, group in itertools.groupby(codeConstrPairs, operator.itemgetter(0))]

    # Step 3: getting only the types in the chunked lists
    chunkedTypes: List[List[ManagedProperties]] = list(map(lambda lst : list(map(lambda pair : pair[1], lst)), chunkedPairs))

    return chunkedTypes
# %%
from sympy.core.numbers import NegativeOne, Number

# Constructors
cs = [Inverse, Transpose, Transpose, Inverse, Inverse, Inverse, Transpose, MatrixSymbol, Symbol, Symbol, Symbol, NegativeOne, NegativeOne, NegativeOne, NegativeOne, Inverse, Symbol, Transpose, Inverse, Symbol, Transpose, Inverse, Inverse, MatMul, MatMul, MatAdd]

res = chunkInvTrans(cs)
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



def stack(byType: ManagedProperties, constructors: List[ManagedProperties]) -> List[ManagedProperties]:
    '''Given a type (like Transpose) and given a list, this function pulls all the signal types to the end of the list, leaving the non-signal-types as the front'''

    # Get number of signal types in the list
    countTypes: int = len(list(filter(lambda c : c == byType, constructors)))

    # Create the signal types that go at the end
    endTypes: List[ManagedProperties] = [byType] * countTypes

    # Get the list without the signal types
    nonSignalTypes: List[ManagedProperties] = list(filter(lambda c : c != byType, constructors))

    return endTypes + nonSignalTypes # + endTypes

assert stack(Transpose, [Inverse, Inverse, Transpose, Inverse, Transpose, Transpose, Inverse, MatMul, MatAdd]) == [Transpose, Transpose, Transpose, Inverse, Inverse, Inverse, Inverse, MatMul, MatAdd]



# %%
def inner(expr): 
    '''Gets the innermost expression (past all the .arg) on the first level only'''
    #isMatSym = lambda e : len(e.free_symbols) == 1

    # TODO missing any base case possibilities? Should include here anything that is not Trace / Inverse ... etc or any kind of constructors that houses an inner argument. 
    Constr = expr.func 
    types = [MatMul, MatAdd, MatrixSymbol, Symbol, Number] 
    isAnySubclass = any(map(lambda t : issubclass(Constr, t), types))

    if (Constr in types) or isAnySubclass: 

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
# NOTE: expr is from test 14 a
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


def transposeOut_Simple_MatMul_or_MatSym(expr: MatrixExpr) -> MatrixExpr: 
    '''For each layered (nested) expression where transpose is the inner operation, this function brings transposes to be the outer operations, leaving all other operations in between in the same order.'''


    ps = list(preorder_traversal(expr)) # elements broken down
    cs = list(map(lambda p: type(p), ps)) # types / constructors


    # Check first: does expr have transpose or inverse? If not, then return it out, nothing to do here: 
    if not (Transpose in cs): 
        return expr 

    csChunked = chunkInvTrans(cs)

    # Order the types properly now for each chunk: make transposes go last in each chunk: 
    stackedChunks = list(map(lambda lst : stack(Transpose, lst), csChunked))


    # Get the lengths of each chunk
    chunkLens = list(map(lambda cLst : len(cLst), csChunked))


    # Use lengths to segregate the preorder traversal exprs also, then later to apply the transformations
    psChunked = []
    rest = ps 

    for size in chunkLens:
        (fst, rest) = (rest[:size], rest[size: ]) 
        psChunked.append( fst )


    def applyTypesToExpr( pairTypeExpr: Tuple[List[ManagedProperties], MatrixExpr]) -> MatrixExpr:

        (typeList, expr) = pairTypeExpr 
        return compose(*typeList)(expr)



    # Pair up the correct order of transpose types with the expressions
    itListExprListPair = list(filter(lambda plst : ((Transpose in plst[0]) or (Inverse in plst[0]) ) , list(zip(stackedChunks, psChunked)) ))


    # Get the first expression only, since it is the most layered, don't use the entire expression list
    itListExprPair = list(map(lambda tsPs : (tsPs[0], tsPs[1][0]), itListExprListPair))

    # Get the inner argument (lay bare) in preparation for apply the correct order of transpose types. 
    itListInnerExprPair = list(map(lambda tsPs : (tsPs[0], inner(tsPs[1])), itListExprPair))

    outs = list(map(lambda tsExprPair : applyTypesToExpr(tsExprPair), itListInnerExprPair ))

    # Get the original expressions as they were before applying correct transpose
    ins = list(map(lambda tsPs : tsPs[1], itListExprPair))

    # Filter: get just the matmul-type arguments (meaning not the D^T or E^-1 type arguments) from the result list (assuming there are other MatMul exprs). Could have done this when first filtering the psChunked, but easier to do it now. 
    # NOTE: when there are ONLY mat syms and no other matmuls then we must keep them since it means the expression is layered with only a matsum as the innermost expression, rather than a matmul. 
    isSymOrNum = lambda expr : expr.is_Symbol or expr.is_Number 
    #isSym = lambda expr  : len(expr.free_symbols) == 1
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

    resultTranspOut = foldLeft(f , accFirst,  outInPairs) # outsNotsPairs[1:])

    return resultTranspOut 



def transposeOut_Simple(expr: MatrixExpr) -> MatrixExpr: 
    
    '''Brings transposes to the outermost level when in a nested expression. Leaves the nested expressions in their same structure. '''

    #isMatSym = lambda s: len(s.free_symbols) == 1

    if not expr.is_MatAdd: 
        return transposeOut_Simple_MatMul_or_MatSym(expr)

    addendsOut: List[MatrixExpr] = list(map(lambda a: transposeOut_Simple(a), expr.args))

    return MatAdd(*addendsOut)

# %% -------------------------------------------------------------




# TEST 1: inverse out, transpose in

B = MatrixSymbol("B", c, c)
C = MatrixSymbol('C', c, c)
E = MatrixSymbol('E', c, c)

expr = Inverse(Transpose(C*E*B))

# %%
res = groupTranspose(expr)
check = expr 

showGroup([
    expr, res, check 
])

assert equal(res, check)
assert equal(expr.doit(), res.doit())
# %%


res = transposeOut_Simple(expr)
check = Transpose(Inverse(MatMul(C, E, B)))

showGroup([
    expr, res, check
])

assert equal(res, check)
assert equal(expr.doit(), res.doit())
# %% -------------------------------------------------------------


# TEST 2: transpose out, inverse in

B = MatrixSymbol("B", c, c)
C = MatrixSymbol('C', c, c)
E = MatrixSymbol('E', c, c)

expr = Transpose(Inverse(C*E*B))

# %%
res = groupTranspose(expr)
check = expr 

showGroup([
    expr, res, check 
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %%
res = transposeOut_Simple(expr)
check = Transpose(Inverse(C*E*B))

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %% -------------------------------------------------------------


# TEST 3: individual transposes inside inverse

B = MatrixSymbol("B", c, c)
C = MatrixSymbol('C', c, c)
E = MatrixSymbol('E', c, c)

expr = Inverse(B.T * E.T * C.T)
# %%

res = groupTranspose(expr)
check = Inverse(Transpose(C*E*B))

showGroup([
    expr, res, check 
])
assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %%

res = transposeOut_Simple(expr)
check = expr 

#checkAggr = Transpose(Inverse(C*E*B))

showGroup([
    expr, res, check
])
assert equal(expr.doit(), res.doit())
assert equal(check, res)
# %% -------------------------------------------------------------


# TEST 4: individual inverse inside transpose

B = MatrixSymbol("B", c, c)
C = MatrixSymbol('C', c, c)
E = MatrixSymbol('E', c, c)

expr = Transpose(B.I * E.I * C.I)
# %%

res = groupTranspose(expr)
check = expr

showGroup([
    expr, res, check 
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)

# %%

res = transposeOut_Simple(expr)
check = expr

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %% -------------------------------------------------------------


# TEST 5 a: individual symbols

A = MatrixSymbol("A", a, b)
C = MatrixSymbol('C', c, c)


# %%

(expr1, check1) = (A, A)
(expr2, check2) = (A.T, A.T) 
(expr3, check3) = (C.I, C.I) 
(expr4, check4) = (Inverse(Transpose(C)), Inverse(Transpose(C)))

res1 = groupTranspose(expr1)
res2 = groupTranspose(expr2)
res3 = groupTranspose(expr3)
res4 = groupTranspose(expr4)

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


(expr1, check1) = (A, A)
(expr2, check2) = (A.T, A.T) 
(expr3, check3) = (C.I, C.I) 
(expr4, check4) = (Inverse(Transpose(C)), Transpose(Inverse(C)))

res1 = transposeOut_Simple(expr1)
res2 = transposeOut_Simple(expr2)
res3 = transposeOut_Simple(expr3)
res4 = transposeOut_Simple(expr4)

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

A = MatrixSymbol("A", c, c)
C = MatrixSymbol('C', c, c)
R = MatrixSymbol('R', c, c)
M = MatrixSymbol('M', c, c)

expr = Transpose(Inverse(Inverse(Inverse(Transpose(MatMul(
    A, 
    Inverse(Transpose(Inverse(Transpose(C)))),
    Inverse(Transpose(R)), 
    M
))))))
# %%

res = groupTranspose(expr)


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


res = transposeOut_Simple(expr)

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

A = MatrixSymbol('A', a, a)
B = MatrixSymbol("B", a, a)
R = MatrixSymbol('R', a, a)
J = MatrixSymbol('J', a, a)

expr = MatMul( Transpose(A*B), Transpose(R*J) )
# %%

res = groupTranspose(expr)
check = Transpose(R*J*A*B)

showGroup([
    expr, res, check 
])

assert equal(expr.doit(), res.doit())
# TODO if you want you should split the matmul case into a "simple matmul case" in that there are no innermost nestings and everything is on the first level. 
assert equal(res, check)

# %%

res = transposeOut_Simple(expr)
check = expr 

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %% -------------------------------------------------------------


# TEST 7: individual transposes littered along as matmul

A = MatrixSymbol('A', a, a)
B = MatrixSymbol("B", a, a)
R = MatrixSymbol('R', a, a)
J = MatrixSymbol('J', a, a)

expr = B.T * A.T * J.T * R.T

# %%

res = groupTranspose(expr)
check = Transpose(R*J*A*B)

showGroup([
    expr, res, check 
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %%

res = transposeOut_Simple(expr)
check = expr

showGroup([expr, res, check])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %% -------------------------------------------------------------


# TEST 8: inverses mixed with transpose in a matmul, but with transposes all as the outer expression

A = MatrixSymbol('A', a, a)
B = MatrixSymbol("B", a, a)
L = MatrixSymbol('L', a, a)
K = MatrixSymbol('K', a, a)
E = MatrixSymbol('E', a, a)
R = MatrixSymbol('R', a, a)

expr = MatMul(A , Transpose(Inverse(R)), Transpose(Inverse(L)) , K , E.T , B.I )
# %%

res = groupTranspose(expr)

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

res = transposeOut_Simple(expr)

check = expr 

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)


# %% -------------------------------------------------------------


# TEST 9: mix of inverses and transposes in a matmul, but this time with transpose not as outer operation, for at least one symbol case. 

A = MatrixSymbol('A', a, a)
B = MatrixSymbol("B", a, a)
R = MatrixSymbol('R', a, a)
L = MatrixSymbol('L', a, a)
K = MatrixSymbol('K', a, a)
E = MatrixSymbol('E', a, a)


expr = MatMul(A , Transpose(Inverse(R)), Inverse(Transpose(L)) , K , E.T , B.I )
# %%

res = groupTranspose(expr)

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

res = transposeOut_Simple(expr)

check = MatMul(
    A, Transpose(Inverse(R)), Transpose(Inverse(L)), K, E.T, B.I
)

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %% -------------------------------------------------------------


# TEST 10: transposes in matmuls and singular matrix symbols, all in a matadd expression. 

A = MatrixSymbol('A', a, a)
B = MatrixSymbol("B", a, a)
R = MatrixSymbol('R', a, a)
L = MatrixSymbol('L', a, a)
K = MatrixSymbol('K', a, a)
E = MatrixSymbol('E', a, a)
D = MatrixSymbol('D', a, a)

expr = A * R.T * L.T * K * E.T * B + D.T + K
# %%

res = groupTranspose(expr)
resGroup = groupTranspose(expr, combineAdds = True)

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

res = transposeOut_Simple(expr)
check = expr

showGroup([
    expr, res, check
])

assert equal(expr.doit(), res.doit())
assert equal(res, check)
# %% -------------------------------------------------------------


# TEST 11: digger case, very layered expression (with transposes separated so not expecting the grouptranspose to change them). Has inner arg matmul. 

C = MatrixSymbol('C', a, a)
E = MatrixSymbol('E', a, a)
D = MatrixSymbol('D', a, a)

expr = Trace(Transpose(Inverse(Transpose(C*D*E))))
# %%

res = groupTranspose(expr)
check = expr 

showGroup([
    expr, res, check 
])

assert equal(expr.doit(), res.doit())
# TODO fix the expandMatMul function so that trace can be passed as argument (just do expand on what is inside)
#assert equal(res, check)
assert res == check
# %%

res = transposeOut_Simple(expr)
check = Trace(Transpose(Transpose(Inverse(C*D*E))))

showGroup([
    expr, res, check
])

# TODO make the expandMatMul function work with Trace (throws error here because Trace has no 'shape' attribute)
# assert equal(expr.doit(), res.doit())
assert expr.doit() == res.doit()
# TODO assert res == check # make structural equality work
assert equal(res, check)
# %% -------------------------------------------------------------



# TEST 12: very layered expression, but transposes are next to each other, with inner arg as matmul

C = MatrixSymbol('C', a, a)
E = MatrixSymbol('E', a, a)
D = MatrixSymbol('D', a, a)

expr = Trace(Transpose(Transpose(Inverse(C*D*E))))

res1 = groupTranspose(expr)

res2 = transposeOut_Simple(expr)

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

C = MatrixSymbol('C', a, a)
A = MatrixSymbol('A', a, a)
D = MatrixSymbol('D', a, a)

expr = Transpose(Inverse(Transpose(C.T * A.I * D.T)))
# %%


res = groupTranspose(expr)

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

res = transposeOut_Simple(expr)

check = Transpose(Transpose(Inverse(
    C.T * A.I * D.T
)))

showGroup([
    expr, res, check
])

assert expr.doit() == res.doit()
assert equal(res, check)
# %% -------------------------------------------------------------



# TEST 14 a: Layered expression (many innermost nestings). The BAR expression has transpose outer and inverse inner, while the other transposes are outer already. (ITIT)

C = MatrixSymbol('C', c, c)
D = MatrixSymbol('D', c, c)
E = MatrixSymbol('E', c, c)
A = MatrixSymbol('A', c, c)
B = MatrixSymbol('B', c, c)
R = MatrixSymbol('R', c, c)
J = MatrixSymbol('J', c, c)


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

res = groupTranspose(expr)

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


res = transposeOut_Simple(expr)

# Aggressive check
#check = Transpose(Transpose(Transpose(Inverse(MatMul(
#    A, D.T, R.T, 
#    Inverse(MatMul(B, Inverse(B*A*R), Transpose(Inverse(E)), C))
#)))))
check = expr

showGroup([
    expr, res, check
])

assert expr.doit() == res.doit()
assert equal(res, check)



# %% -------------------------------------------------------------




# TEST 14 b: layered expression (many innermost nestings). The BAR expression has transpose inner and inverse outer and other transposes are outer already, after inverse. (ITIT)

C = MatrixSymbol('C', c, c)
D = MatrixSymbol('D', c, c)
E = MatrixSymbol('E', c, c)
A = MatrixSymbol('A', c, c)
B = MatrixSymbol('B', c, c)
R = MatrixSymbol('R', c, c)
J = MatrixSymbol('J', c, c)


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

res = groupTranspose(expr)

check = Transpose(Inverse(Transpose(MatMul(
    Inverse(Transpose(MatMul(
        B, Transpose(Inverse(Transpose(B*A*R))), Transpose(Inverse(E)), C
    ))), 
    Transpose(Transpose(MatMul(R, D, A.T)))
))))

showGroup([expr, res, check])

assert equal(res.doit(), expr.doit())
assert equal(res, check)



# %%


res = transposeOut_Simple(expr)

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

C = MatrixSymbol('C', c, c)
D = MatrixSymbol('D', c, c)
E = MatrixSymbol('E', c, c)
A = MatrixSymbol('A', c, c)
B = MatrixSymbol('B', c, c)
R = MatrixSymbol('R', c, c)
J = MatrixSymbol('J', c, c)


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

res = groupTranspose(expr)

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


res = transposeOut_Simple(expr)

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

C = MatrixSymbol('C', c, c)
D = MatrixSymbol('D', c, c)
E = MatrixSymbol('E', c, c)
A = MatrixSymbol('A', c, c)
B = MatrixSymbol('B', c, c)
R = MatrixSymbol('R', c, c)
J = MatrixSymbol('J', c, c)


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

res = groupTranspose(expr)

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


res = transposeOut_Simple(expr)

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



# TEST 15: very layered expression, and the inner arg is matadd with some matmuls but some matmuls have one of the elements as another layered arg (any of the test 14 cases, like 14 a), so we can test if the function reaches all rabbit holes effectively. 

C = MatrixSymbol('C', c, c)
D = MatrixSymbol('D', c, c)
E = MatrixSymbol('E', c, c)
A = MatrixSymbol('A', c, c)
B = MatrixSymbol('B', c, c)
R = MatrixSymbol('R', c, c)
J = MatrixSymbol('J', c, c)

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

res = groupTranspose(expr)

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


res = transposeOut_Simple(expr)

# Simple check 14 a
check_inner_14a = Inverse(
    MatMul(A*D.T*R.T , 
        Transpose(
            Inverse(
                MatMul(C.T, E.I, L, B.T)
            ) 
        )
    )
)

# Aggressive check 14 a
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
    # First check if not a matrix symbol; if it is then diff easily. 
    if isinstance(expr, MatrixSymbol): 
        return diff(Trace(expr), byVar)

    # Check now that it is matmul
    assert expr.is_MatMul
    
    # Get how many of byVar are in the expression: 
    # NOTE: arg may be under Transpose or Inverse here, not just a simple MatrixSymbol, so we need to detect the MatrixSymbol underneath using "has" test instead of the usual == test of arg == byVar. 
    # This way we count the byVar underneath the Transp or Inverse.
    numSignalVars = len(list(filter(lambda arg: arg.has(byVar), expr.args)))
    # NOTE here is a way to get the underlying MatrixSymbol when there are Invs and Transposes (just the letters without the invs and transposes): 
    # list(itertools.chain(*map(lambda a: a.free_symbols, trace.arg.args))), where expr := trace.arg

    # Get the split list applications: split by signal var for how many times it appears
    signalSplits = list(map(lambda n: splitOnce(expr.args, byVar, n), range(1, numSignalVars + 1)))

    # Apply the trace derivative function per pair
    transposedMatMuls = list(map(lambda s: traceDerivPair(s), signalSplits))
    
    # Result is an addition of the transposed matmul combinations:
    return MatAdd(* transposedMatMuls )



# %%



def derivTrace(trace: Trace, byVar: MatrixSymbol) -> MatrixExpr: 
    '''
    Does derivative of a Trace expression. 
    Equivalent to diff(trace, byVar). 
    '''
    assert trace.is_Trace 

    # Case 1: trace of a single matrix symbol - easy
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

        # NOTE this is a list of MatAdds; must flatten them to avoid brackets extra and to enhance simplification.
        diffedAddends: List[MatrixExpr] = list(map(lambda m : derivMatInsideTrace(m, byVar), addends))

        # Preparing to flatten the matrix additions into one overall matrix addition: 
        splitMatAdd = lambda expr : list(expr.args) if expr.is_MatAdd else [expr]

        # Splitting and flattening here: 
        splitDiffedAddends = list(itertools.chain(*map(lambda d : splitMatAdd(d), diffedAddends)) ) 

        # Now return the mat add
        return MatAdd(*splitDiffedAddends)

    



# %% -------------------------------------------------------------




### TEST 1: simple case, with addition, no inverse or transpose in any of the variables
a, b, c = symbols('a b c', commutative=True)

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
    groupTranspose(res), 
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
    groupTranspose(res)
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
    res, check, dcheck, groupTranspose(dcheck)
])
# %%
assert equal(res, check)
assert equal(res, dcheck)
assert equal(check, dcheck)

showGroup([
    res, 
    check, 
    dcheck, 
    groupTranspose(res)
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
# matrixDifferential(a*X, X)
# %%
# TODO error with cyclic permute
# # matrixDifferential(Trace(R_), R_)
# %%
# matrixDifferential(A*J, A)

