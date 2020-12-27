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
from sympy.matrices.expressions.matpow import MatPow 

from sympy.core.numbers import NegativeOne, Number

from sympy.core.assumptions import ManagedProperties


# %%
import torch

# Types
MatrixType = ManagedProperties

Tensor = torch.Tensor
LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor

# %% codecell
import sys

PATH: str = '/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP'

UTIL_DISPLAY_PATH: str = PATH + "/src/utils/GeneralUtil/"

MATDIFF_PATH: str = PATH + "/src/MatrixCalculusStudy/DifferentialLib"


sys.path.append(PATH)
sys.path.append(UTIL_DISPLAY_PATH)
sys.path.append(MATDIFF_PATH)

from src.MatrixCalculusStudy.DifferentialLib.printingLatex import myLatexPrinter


# For displaying
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax', latex_printer= lambda e, **kw: myLatexPrinter.doprint(e))


from src.MatrixCalculusStudy.DerivativeLib.main.Simplifications import *
from src.MatrixCalculusStudy.DerivativeLib.test.utils.TestHelpers import * 


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
from src.MatrixCalculusStudy.DifferentialLib.simplifications import _conditional_replace, cyclic_permute_dX_cond, cyclic_permute_dX_repl

# %%
# STRATEGY 3: DIFFERENTIAL APPROACH -------------------------------

from src.MatrixCalculusStudy.DifferentialLib.symbols import d

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











# %% -------------------------------------------------------------


### TRACE DERIVATIVE HERE


# Now apply the trace deriv function per pair
def calcTraceDerivFromPair(useConstr: MatrixType , pair: ArgPair, signalVar: MatrixSymbol) -> MatrixExpr:

    #def combiner(left: List[MatrixExpr], right: List[MatrixExpr], useConstr: MatrixType, signalVar: MatrixSymbol):

    def combineWhenTransposeArg(left: List[MatrixExpr], right: List[MatrixExpr]) -> MatrixExpr:

        result = MatMul(*list(itertools.chain(*(right, left))))
        return result 


    def combineWhenInverseArg(left: List[MatrixExpr], right: List[MatrixExpr], signalVar: MatrixSymbol) -> MatrixExpr:

        # Create the inner matmul (that is inside the inverse var)
        middle: MatrixExpr = MatMul(* list(itertools.chain(* (right, left) )))
        # Create the inverse diff expression result
        invVar = Inverse(signalVar)
        result = MatMul(NegativeOne(-1), Transpose(MatMul(invVar, middle, invVar)) )

        return result 



    def combineWhenSimpleArg(left: List[MatrixExpr], right: List[MatrixExpr]) -> MatrixExpr:
        def putTranspInList(lstSyms: List[MatrixSymbol]) -> List[MatrixExpr]:
            if len(lstSyms) == 0:
                return []
            elif len(lstSyms) == 1: 
                return [Transpose(*lstSyms)]
            #else: 
            return [Transpose(MatMul(*lstSyms))]
        
        result = MatMul(* (putTranspInList(left) + putTranspInList(right)) )

        return result 
        
    
    (left, right) = pair

    if left == [] and right == []:
        # TODO this function is not prepared for when both are empty (what else to return but exception? WANT OPTION MONAD NOW)
        raise Exception("Both `left` and `right` are empty")


    if useConstr == Transpose:
        return combineWhenTransposeArg(left, right)
    elif useConstr == Inverse:
        return combineWhenInverseArg(left, right, signalVar = signalVar)
    #else: is matrixsymbol
    # TODO what is good practice, need to assert useConstr == MatrixSymbol? how to be more typesafe????
    return combineWhenSimpleArg(left, right)


# %%
def calcDerivInsideTrace(expr: MatrixExpr, byVar: MatrixSymbol) -> MatMul:
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
    signalSplits: List[Tuple[MatrixType, ArgPair]] = splitArgs(givenArgs = expr.args, signalVar = byVar)

    # Apply the trace derivative function per pair
    components = list(map(lambda typeAndPair: calcTraceDerivFromPair(*typeAndPair, signalVar = byVar), signalSplits))

    # Result is an addition of the transposed matmul combinations:
    return MatAdd(* components )



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
        return calcDerivInsideTrace(trace.arg, byVar = byVar)
        #assert equal(result, diff(trace, byVar))

    # Case 3: split first by MatAdd to get MatMul pieces and feed in the pieces to the single function that gets applied to each MatMul piece.
    elif trace.arg.is_MatAdd:
        # Filter the matrixsymbols that are byVar and the matrixexprs that contain the byVar
        addends: List[MatrixExpr] = list(filter(lambda m : m.has(byVar), trace.arg.args ))
        # NOTE: can contain matrixsymbols mixed with matmul

        # NOTE this is expr list of MatAdds; must flatten them to avoid brackets extra and to enhance simplification.
        diffedAddends: List[MatrixExpr] = list(map(lambda m : calcDerivInsideTrace(m, byVar), addends))

        # Preparing to flatten the matrix additions into one overall matrix addition:
        splitMatAdd = lambda expr : list(expr.args) if expr.is_MatAdd else [expr]

        # Splitting and flattening here:
        splitDiffedAddends = list(itertools.chain(*map(lambda d : splitMatAdd(d), diffedAddends)) )

        # Now return the mat add
        return MatAdd(*splitDiffedAddends)



# %%

# TRACE DERIVATIVE TESTS: 
a, b, c = symbols('a b c', commutative=True)

A = MatrixSymbol("A", c, c)
R = MatrixSymbol("R", c, c)
J = MatrixSymbol("J", c, c)
C = MatrixSymbol('C', c, c)
E = MatrixSymbol('E', c, c)
B = MatrixSymbol('B', c, c)
L = MatrixSymbol('L', c, c)
D = MatrixSymbol('D', c, c)

# %% --------------------------------------------------------------


### TRACE TEST 1: simple case, with addition, no inverse or transpose in any of the variables

trace = Trace( A*B*E * R * J  + C*E*B*E*L*E*D )
byVar = E

groupedCheck = MatAdd(
    MatMul(Transpose(MatMul(A, B)), Transpose(MatMul(R, J))),
    MatMul(C.T, Transpose(MatMul(B, E, L, E, D))),
    MatMul(Transpose(MatMul(C, E, B)), Transpose(MatMul(L, E, D))),
    MatMul(Transpose(MatMul(C, E, B, E, L)), D.T)
)

onlineCheck = Transpose(MatAdd(
    MatMul(R,J,A,B),
    MatMul(B,E,L,E,D,C),
    MatMul(L,E,D,C,E,B),
    MatMul(D,C,E,B,E,L)
))

testDerivAlgo(algo = derivTrace, expr = trace, byVar = byVar, groupedCheck = groupedCheck, onlineCheck = onlineCheck)

# %% -------------------------------------------------------------


# TRACE TEST 2a: testing one inverse expression, not the byVar
trace = Trace(C * E * B * E * Inverse(L) * E * D)
byVar = E

groupedCheck = Transpose(MatAdd(
    B * E * Inverse(L) * E * D * C,
    Inverse(L)*E*D*C*E*B,
    D*C*E*B*E*Inverse(L)
)) 

onlineCheck = groupedCheck 

testDerivAlgo(algo = derivTrace, expr = trace, byVar = byVar, groupedCheck = groupedCheck, onlineCheck = onlineCheck)

# %% -------------------------------------------------------------


# TRACE TEST 2b: testing one inverse expression, not the byVar

trace = Trace(B * Inverse(C) * E * L * A * E * D)
byVar = E

groupedCheck = Transpose(L * A* E * D * B * Inverse(C) + D*B*Inverse(C)*E*L*A)
onlineCheck = groupedCheck

testDerivAlgo(algo = derivTrace, expr = trace, byVar = byVar, groupedCheck = groupedCheck, onlineCheck = onlineCheck)

# %% -------------------------------------------------------------


# TRACE TEST 2c: testing one inverse expression, not the byVar, that is situated at the front of the expression.

trace = Trace(Inverse(C) * E * B * E * L * E * D)
byVar = E

groupedCheck = Transpose(B*E*L*E*D*Inverse(C) + L*E*D*Inverse(C)*E*B + D*Inverse(C)*E*B*E*L)
onlineCheck = groupedCheck

testDerivAlgo(algo = derivTrace, expr = trace, byVar = byVar, groupedCheck = groupedCheck, onlineCheck = onlineCheck)
# %% -------------------------------------------------------------


# TRACE TEST 3a: testing mix of inverse and transpose expressions, not the byVar

trace = Trace(B * C.I * E * L.T * A * E * D)
byVar = E

groupedCheck = MatAdd(
    MatMul(Transpose(B * C.I), Transpose(L.T * A * E* D)),
    MatMul(Transpose(B * C.I * E * L.T * A), Transpose(D))
)

onlineCheck = MatAdd(
    MatMul(Transpose(A*E*D*B*Inverse(C)), L),
    MatMul(A.T, L, E.T, Transpose(Inverse(C)), B.T, D.T)
)

testDerivAlgo(algo = derivTrace, expr = trace, byVar = byVar, groupedCheck = groupedCheck, onlineCheck = onlineCheck)
# %% -------------------------------------------------------------


### TRACE TEST 3b: testing mix of inverser and transpose expressions, and byVar is either an inverse of transpose.

trace = Trace(B * C.I * E.T * L.T * A * E * D)
byVar = E

groupedCheck = MatAdd(
    MatMul(L.T,A,E,D,B,C.I),
    MatMul(Transpose(B*C.I*E.T * L.T*A), Transpose(D))
)
onlineCheck = L.T * A*E*D*B*C.I + A.T* L * E *(C.I).T * B.T * D.T


testDerivAlgo(algo = derivTrace, expr = trace, byVar = byVar, groupedCheck = groupedCheck, onlineCheck = onlineCheck)

# %% -------------------------------------------------------------


### TRACE TEST 4a: no inverse or transpose, but the byVar is at the edges and in the middle

trace = Trace(E * C * A * B * E * L * E * D * E)
byVar = E

groupedCheck = MatAdd(
    Transpose(C*A*B*E*L*E*D*E),
    MatMul(Transpose(E*C*A*B), Transpose(L*E*D*E)),
    MatMul(Transpose(E*C*A*B*E*L), Transpose(D*E)),
    Transpose(E*C*A*B*E*L*E*D)
)

onlineCheck = Transpose(MatAdd(
    MatMul(C,A,B,E,L,E,D,E),
    MatMul(L,E,D,E,E,C,A,B),
    MatMul(D,E,E,C,A,B,E,L),
    MatMul(E,C,A,B,E,L,E,D)
))

testDerivAlgo(algo = derivTrace, expr = trace, byVar = byVar, groupedCheck = groupedCheck, onlineCheck = onlineCheck)

# %% ---------------------------------------------------------------


### TRACE TEST 4b: no inverse or transpose but the byVar is at the edge and middle (at the left edge, it is wrapped in Transpose)

trace = Trace(E.T * C * A * B * E * L * E * D * E)
byVar = E 

groupedCheck = MatAdd(
    MatMul(C, A, B, E, L, E, D, E),
    MatMul(Transpose(E.T * C*A*B), Transpose(L*E*D*E)),
    MatMul(Transpose(E.T* C*A*B*E*L), Transpose(D*E)),
    Transpose(E.T * C*A*B*E*L*E*D)
)

onlineCheck = MatAdd(
    MatMul(C,A,B,E,L,E,D,E),
    MatMul(Transpose(C*A*B), E, E.T, D.T, E.T, L.T),
    MatMul(Transpose(C*A*B*E*L), E,E.T, D.T),
    MatMul(Transpose(C*A*B*E*L*E*D), E)
)

testDerivAlgo(algo = derivTrace, expr = trace, byVar = byVar, groupedCheck = groupedCheck, onlineCheck = onlineCheck)

# %% --------------------------------------------------------------


### TRACE TEST 4c: no inverse or transpose on other symbols but the bYVar is at the edge and middle and wrapped in either transpose or inverse (shuffle the constrs).

trace = Trace(E.T * C * E * B * E.I * L * E.T * D * E.I)
byVar = E 

groupedCheck = MatAdd(
    MatMul(C,E,B,E.I,L,E.T,D,E.I),
    MatMul(Transpose(E.T * C), Transpose(B * E.I *L*E.T*D*E.I)),
    MatMul(NegativeOne(-1), Transpose(MatMul(
        E.I,  
        MatMul(L, E.T, D, E.I, E.T, C, E, B),  
        E.I
    ))),
    MatMul(D, E.I, E.T, C, E, B, E.I, L),
    MatMul(NegativeOne(-1), Transpose(MatMul(
        E.I, 
        MatMul(E.T, C, E, B, E.I, L, E.T, D), 
        E.I
    )))
)

onlineCheck = MatAdd(
    C* E * B * E.I * L * E.T * D * E.I,
    C.T * E * Transpose(Inverse(E)) * D.T * E * L.T * Transpose(Inverse(E)) * B.T, 
    MatMul(NegativeOne(-1), MatMul(
        Transpose(C*E*B*E.I), 
        MatMul(
            E, Transpose(Inverse(E)), D.T, E, L.T, Transpose(Inverse(E))
        )
    )), 
    D * E.I * E.T * C * E * B * E.I * L, 
    MatMul(NegativeOne(-1), MatMul(
        Transpose(D * Inverse(E)), E, L.T, Transpose(Inverse(E)), B.T, E.T, C.T, E, Transpose(Inverse(E))
    ))
)

testDerivAlgo(algo = derivTrace, expr = trace, byVar = byVar, groupedCheck = groupedCheck, onlineCheck = onlineCheck)



# %%
res = derivTrace(trace, byVar)
polarize(Transpose, res)
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

