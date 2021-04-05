

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

from sympy.core.singleton import Singleton 
from sympy.core.numbers import NegativeOne, Number, Integer, One, IntegerConstant

from sympy.core.assumptions import ManagedProperties

from sympy import UnevaluatedExpr , parse_expr 

# Types
import inspect # for isclass() function 
import collections  # for namedtuple


# Path settings
import sys
import os

PATH: str = '/'

UTIL_DISPLAY_PATH: str = PATH + "/src/utils/GeneralUtil/"

MATDIFF_PATH: str = PATH + "/src/MatrixCalculusStudy/DifferentialLib"


sys.path.append(PATH)
sys.path.append(UTIL_DISPLAY_PATH)
sys.path.append(MATDIFF_PATH)




from src.utils.GeneralUtil import *

# For displaying
from IPython.display import display, Math
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax')





# ---------------------------------------------------------------

### WARNING: not maintained anymore 

def group(WrapType: MatrixType, expr: MatrixExpr, combineAdds:bool = False) -> MatrixExpr:
    '''Combines transposes when they are the outermost operations.

    Brings the individual transpose operations out over the entire group (factors out the transpose and puts it as the outer operation).
    NOTE: This happens only when the transpose ops are at the same level. If they are nested, that "bringing out" task is left to the rippletranspose function.

    combineAdds:bool  = if True then the function will group all the addition components under expr transpose, like in matmul, otherwise it will group only the individual components of the MatAdd under transpose.'''


    def revAlgo(WrapType: MatrixType, expr: MatrixExpr):
        '''Converts B.T * A.T --> (A * B).T'''

        if hasType(WrapType, expr) and isMul(expr):

            revs = list(map(lambda a : pickOut(WrapType, a), reversed(expr.args)))

            #return Transpose(MatMul(*revs))
            return WrapType(MatMul(*revs))

        return expr


    def algo_Group_MatMul_or_MatSym(WrapType: MatrixType, expr: MatrixExpr) -> MatrixExpr:

        ps = list(preorder_traversal(expr))
        #ms = list(filter(lambda p : p.is_MatMul, ps))
        ms = list(filter(lambda p : isMul(p), ps))
        ts = list(map(lambda m : revAlgo(WrapType, m), ms))

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

    # Group function here -----------------------------------

    #if not expr.is_MatAdd: # TODO must change this to identify matmul or matsym exactly
    if not isAdd(expr):
        return algo_Group_MatMul_or_MatSym(WrapType, expr)


    Constr = getConstr(expr)

    # TODO fix this to handle any non-add operation upon function entry
    addendsTransp: List[MatrixExpr] = list(map(lambda a: group(WrapType, a), expr.args))

    if combineAdds:
        innerAddends = list(map(lambda t: pickOut(WrapType, t), addendsTransp))
        #return Transpose(MatAdd(*innerAddends))
        return WrapType (Constr(*innerAddends))

    # Else not combining adds, just grouping transposes in addends individually
    # TODO fix this to handle any non-add operation upon function entry. May not be a MatAdd, may be Trace for instance.

    #return MatAdd(*addendsTransp)
    return Constr(*addendsTransp)



# ---------------------------------------------


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


# NOTE: the symbols can be E.T or E or E.I (so are technically MatrixExpr's not MatrixSymbols)


ArgPair = Tuple[List[MatrixSymbol], List[MatrixSymbol]]

def splitArgs(givenArgs: List[MatrixExpr], signalVar: MatrixSymbol) -> List[Tuple[MatrixType, ArgPair]]:

    '''Splits even at the signalVar if it is wrapped in theArgs with a constructor, and returns a dict specifying under which constructor it was wrapped in. 
    
    For instance if we split A * E.T * B at signalVar = E then result is {Transpose : ([A], [B]) } because the signalVar was wrapped in a transpose. 
    
    Constructor == Transpose ---> dict key is Transpose
    Constructor == Inverse ---> dict key is Inverse
    Constructor == MatrixSymbol --> dict key is MatrixSymbol (or None?)
    '''
    theArgs = list(givenArgs)

    return [ ( theArgs[i].func  ,(theArgs[0:i], theArgs[i+1 : ]) ) for i in range(0, len(theArgs)) if theArgs[i].has(signalVar)] 


# SPLIT ONCE TESTS:

L = MatrixSymbol('L', c, c)

expr = C*E*B*E*L*E*D

assert splitOnce(expr.args, E, 1) == ([C], [B, E, L, E, D])
assert splitOnce(expr.args, E, 2) == ([C, E, B], [L, E, D])
assert splitOnce(expr.args, E, 3) == ([C, E, B, E, L], [D])
assert splitOnce(expr.args, E, 0) == ([], [])
assert splitOnce(expr.args, E, 4) == ([], [])
# TODO how to assert error for negative number n?

