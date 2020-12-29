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
from sympy.core.mul import Mul 
from sympy.core.add import Add
from sympy.core.power import Pow

from sympy.core.numbers import NegativeOne, Number, Integer

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
    #assert expr.is_MatMul
    # TODO should I use isMul to account for output from freeze()?
    assert isMul(expr)

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
    # TODO accounting for output from freeze() here using isMul
    elif isMul(trace.arg):
    #elif trace.arg.is_MatMul:
        return calcDerivInsideTrace(trace.arg, byVar = byVar)
        #assert equal(result, diff(trace, byVar))

    # Case 3: split first by MatAdd to get MatMul pieces and feed in the pieces to the single function that gets applied to each MatMul piece.
    # TODO accounting for output from freeze() using isAdd
    elif isAdd(trace.arg):
    #elif trace.arg.is_MatAdd:
        # Filter the matrixsymbols that are byVar and the matrixexprs that contain the byVar
        addends: List[MatrixExpr] = list(filter(lambda m : m.has(byVar), trace.arg.args ))
        # NOTE: can contain matrixsymbols mixed with matmul

        # NOTE this is expr list of MatAdds; must flatten them to avoid brackets extra and to enhance simplification.
        diffedAddends: List[MatrixExpr] = list(map(lambda m : calcDerivInsideTrace(m, byVar), addends))

        # Preparing to flatten the matrix additions into one overall matrix addition:
        #splitMatAdd = lambda expr : list(expr.args) if expr.is_MatAdd else [expr]
        # TODO accounting for freeze output by using isAdd
        splitMatAdd = lambda expr : list(expr.args) if isAdd(expr) else [expr]

        # Splitting and flattening here:
        splitDiffedAddends = list(itertools.chain(*map(lambda d : splitMatAdd(d), diffedAddends)) )

        # Now return the mat add
        matadd = MatAdd(*splitDiffedAddends)

        # Apply freeze (since this is matadd constructor, needed here so that intermediate components don't get evaluated when shown)
        #mataddFreeze = freeze(matadd)

        #return mataddFreeze
        return matadd 



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
# %%
res = derivTrace(trace, byVar)
p = polarize(Transpose, res.args[1])
p
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
fr = freeze(res)
fr

# %% codecell
