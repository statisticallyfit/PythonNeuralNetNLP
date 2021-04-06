

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




# # TEST 1a: SL + GA = single symbol + grouped MatAdd
expr_SLaGA = MatAdd(A, Transpose(B + C.T) )
check = expr_SLaGA

testSimplifyAlgo(algo = group, expr = expr_SLaGA, check = check, byType = Transpose)

check = Transpose( (B + C.T) + A.T )

testSimplifyAlgo_GroupCombineAdds(expr = expr_SLaGA, check = check, byType = Transpose)

# TEST 1b: SL + GA = single symbol + grouped MatAdd (with more layerings per addend)

expr_SLaGA = MatAdd(
    Inverse(Transpose(Transpose(A))),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        B , Inverse(Transpose(Transpose(E))) , C.T , R
    ))))) )
)


check = MatAdd(
    Inverse(Transpose(Transpose(A))),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        Inverse(Transpose(Transpose(E))), B, R, C.T
    ))))))
)
testSimplifyAlgo(algo = group, expr = expr_SLaGA, check = check, byType = Transpose)


# TEST 2a: SL * GA = single symbol * grouped MatAdd

expr_SLmGA = MatMul(
    A,
    Inverse(Transpose(MatAdd(B, C.T)))
)
# TODO result got too complicated
check = Transpose(MatMul(
    Transpose(Inverse(Transpose(MatAdd(B, C.T)))),
    A.T
))

testSimplifyAlgo(algo = group, expr = expr_SLmGA, check = check, byType = Transpose)




# TEST 2b: SL * GA = single symbol * grouped MatAdd (with more layerings per addend)

expr_SLmGA = MatMul(
    Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))),
    Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) )
)

check = Transpose(MatMul(

    Transpose(Inverse(Transpose(Inverse(Transpose(Transpose(MatAdd(
        B , Inverse(Transpose(Transpose(E))) , C.T , Transpose(Inverse(R))
    ))))) )),
    Transpose(Inverse(Transpose(Transpose(Inverse(Inverse(Transpose(Transpose(A))))))))
))

testSimplifyAlgo(algo = group, expr = expr_SLmGA, check = check, byType = Transpose)




# TEST 3a: SL + GM = single symbol + grouped MatMul

expr_SLaGM = MatAdd(A, MatMul(B, A.T, R.I))
check  = MatAdd(
    A,
    Transpose(MatMul(
        Transpose(Inverse(R)), A, B.T
    ))
)
testSimplifyAlgo(algo = group, expr = expr_SLaGM, check = check, byType = Transpose)

# GENERAL TEST 1: inverse out, transpose in
expr = Inverse(Transpose(C*E*B))

check = expr

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)




# GENERAL TEST 2: transpose out, inverse in
expr = Transpose(Inverse(C*E*B))
check = expr

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)



# GENERAL TEST 3: individual transposes inside inverse
expr = Inverse(B.T * E.T * C.T)
check = Inverse(Transpose(C*E*B))

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)


# GENERAL TEST 4: individual inverse inside transpose

expr = Transpose(B.I * E.I * C.I)

check = expr

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)



# GENERAL TEST 5 a: individual symbols

Q = MatrixSymbol("Q", a, b)

(expr1, check1) = (Q, Q)
(expr2, check2) = (Q.T, Q.T)
(expr3, check3) = (C.I, C.I)
(expr4, check4) = (Inverse(Transpose(C)), Inverse(Transpose(C)))

testSimplifyAlgo(algo = group, expr = expr1, check = check1, byType = Transpose)
testSimplifyAlgo(algo = group, expr = expr2, check = check2, byType = Transpose)
testSimplifyAlgo(algo = group, expr = expr3, check = check3, byType = Transpose)
testSimplifyAlgo(algo = group, expr = expr4, check = check4, byType = Transpose)





# GENERAL TEST 5 b: inidivudal symbols nested

expr = Transpose(Inverse(Inverse(Inverse(Transpose(MatMul(
    A,
    Inverse(Transpose(Inverse(Transpose(C)))),
    Inverse(Transpose(R)),
    M
))))))
check = Transpose(Inverse(Inverse(Inverse(Transpose(Transpose(MatMul(
    M.T,
    Transpose(Inverse(Transpose(R))),
    Transpose(Inverse(Transpose(Inverse(Transpose(C))))),
    A.T
)))))))

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)



# GENERAL TEST 6: grouped products

expr = MatMul( Transpose(A*B), Transpose(R*J) )
check = Transpose(R*J*A*B)

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)


# GENERAL TEST 7: individual transposes littered along as matmul

expr = B.T * A.T * J.T * R.T
check = Transpose(R*J*A*B)

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)



# GENERAL TEST 8: inverses mixed with transpose in expr matmul, but with transposes all as the outer expression
L = Transpose(Inverse(MatMul(B, A, R)))

expr = MatMul(A , Transpose(Inverse(R)), Transpose(Inverse(L)) , K , E.T , B.I )
check = Transpose(
    MatMul( Transpose(Inverse(B)), E , K.T , Inverse(L) , Inverse(R) , A.T)
)

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)




# GENERAL TEST 9: mix of inverses and transposes in expr matmul, but this time with transpose not as outer operation, for at least one symbol case.
L = Inverse(Transpose(MatMul(B, A, R)))

expr = MatMul(A , Transpose(Inverse(R)), Inverse(Transpose(L)) , K , E.T , B.I )
check = Transpose(
    MatMul( Transpose(Inverse(B)), E , K.T , Transpose(Inverse(Transpose(L))) , R.I , A.T)
)
testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)



# GENERAL TEST 10: transposes in matmuls and singular matrix symbols, all in expr matadd expression.
L = Transpose(Inverse(MatMul(B, A, R)))

expr = MatAdd(
    MatMul(A, R.T, L, K, E.T, B),
    D.T, K
)
#check = MatAdd( Transpose(MatMul(B.T, E, K.T, Transpose(L), R, A.T)) , D.T, K)
check = MatAdd(
    Transpose(MatMul(B.T, E, K.T, Inverse(MatMul(B, A, R)), R, A.T)), K, D.T
)

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)


check = Transpose(MatAdd(
    MatMul(B.T, E, K.T, Inverse(MatMul(B, A, R)), R, A.T), D, K.T
))

testSimplifyAlgo_GroupCombineAdds(expr = expr, check = check, byType = Transpose)


# GENERAL TEST 11: digger case, very layered expression (with transposes separated so not expecting the grouptranspose to change them). Has inner arg matmul.

expr = Trace(Transpose(Inverse(Transpose(C*D*E))))
check = expr

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)

testSimplifyAlgo_GroupCombineAdds(expr = expr, check = check, byType = Transpose)



# GENERAL TEST 12: very layered expression, but transposes are next to each other, with inner arg as matmul

expr = Trace(Transpose(Transpose(Inverse(C*D*E))))
check = expr

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)

testSimplifyAlgo_GroupCombineAdds(expr = expr, check = check, byType = Transpose)

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)



# GENERAL TEST 13: very layered expression (digger case) with individual transpose and inverses littered in the inner matmul arg.
expr = Transpose(Inverse(Transpose(C.T * A.I * D.T)))
check = Transpose(Inverse(Transpose(Transpose(
    MatMul(D , Transpose(Inverse(A)), C )
))))

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)




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

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)
testSimplifyAlgo_GroupCombineAdds(expr = expr, check = check, byType = Transpose)




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
check = Transpose(Inverse(Transpose(MatMul(
    Inverse(Transpose(MatMul(
        B, Transpose(Inverse(Transpose(B*A*R))), Transpose(Inverse(E)), C
    ))),
    Transpose(Transpose(MatMul(R, D, A.T)))
))))

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)





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

check = Inverse(Transpose(Transpose(MatMul(
    Transpose(Inverse(Transpose(Transpose(MatMul(
        B, Inverse(B*A*R), Transpose(Inverse(E)), C
    ))))),
    Transpose(Transpose(MatMul(R, D, A.T)))
))))

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)




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
check = Inverse(Transpose(Transpose(MatMul(
    Transpose(Inverse(Transpose(Transpose(MatMul(
        B, Transpose(Inverse(Transpose(B*A*R))), Transpose(Inverse(E)), C
    ))))),
    Transpose(Transpose(MatMul(R, D, A.T)))
))))

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)






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

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)



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

testSimplifyAlgo_GroupCombineAdds(expr = expr, check = check, byType = Transpose)
# TODO: faulty function: group combine add should have gone in the inside of the outer transpose and done transpose combine there.


# GENERAL TEST 16: testing mix and match of matmul / matadd with inverse / transposes to see how polarize filters out Transpose. (Meant for mainly testing the polarize function)

expr_polarize = Inverse(MatMul(
    Transpose(Inverse(Transpose(MatAdd(B.T, A.T, R, MatMul(Transpose(Inverse(B*A*R.T)), MatAdd(E, J, D)), Inverse(Transpose(E)), Inverse(Transpose(D)))))),
    Inverse(Transpose(MatMul(A.T, B.T, E.I, Transpose(Inverse(Transpose(A + E + R.T))), C)))
))
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

testSimplifyAlgo(algo = group, expr = expr_polarize, check = check, byType = Transpose)
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

