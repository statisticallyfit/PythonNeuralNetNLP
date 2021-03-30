
# %%

import inspect 
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

MATDIFF_PATH: str = PATH + "/src/MatrixCalculusStudy/DerivativeLib"


sys.path.append(PATH)
sys.path.append(UTIL_DISPLAY_PATH)
sys.path.append(MATDIFF_PATH)

# For displaying
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax')


from src.MatrixCalculusStudy.DerivativeLib.main.Simplifications import *

from src.MatrixCalculusStudy.DerivativeLib.test.utils.TestHelpers import *




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
X = MatrixSymbol("X", c, c)


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

# %%
# INNER TEST 4: powers

e =  Transpose(MatMul(Transpose(R*J*A.T*B), A*X))
ein = MatAdd(E, Transpose(MatMul(Transpose(R*J*A.T*B), MatPow(X*A, 4))))
eout = MatAdd(E, Transpose(MatMul(
    Transpose(R*J*A.T*B), 
    MatPow(Transpose(J + X*A), 4)
)))
p = Transpose(MatPow(Inverse(MatPow(MatPow(X, 2), 5)), 3))

showGroup([e, ein, eout, p])



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

testSimplifyAlgo(algo = digger, expr = expr, check = check, byType = None)



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



# # TEST 1a: SL + GA = single symbol + grouped MatAdd
expr_SLaGA = MatAdd(A, Transpose(B + C.T) )
# %%
check = expr_SLaGA

testSimplifyAlgo(algo = group, expr = expr_SLaGA, check = check, byType = Transpose)
# %%

check = Transpose( (B + C.T) + A.T )

testSimplifyAlgo_GroupCombineAdds(expr = expr_SLaGA, check = check, byType = Transpose)
# %%
# TODO see the todo for matadd general addends error in the rippleout function
check = expr_SLaGA

testSimplifyAlgo(rippleOut, expr = expr_SLaGA, check = expr_SLaGA, byType = Transpose)

# %%

check = expr_SLaGA

testSimplifyAlgo(algo = factor, expr = expr_SLaGA, check = check, byType = Transpose)
# %%

# NOTE not liking the polarize result here - got too complicated. # Fixed with countTopTransp inner function inside polarize()

#check_TOFIX = Transpose(MatAdd(
#    Transpose(MatAdd(C, B.T)), A.T
#))
#check = expr_SLaGA
check = MatAdd(A, MatAdd(C, B.T))

testSimplifyAlgo(algo = polarize, expr = expr_SLaGA, check = check, byType = Transpose)

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
testSimplifyAlgo(algo = group, expr = expr_SLaGA, check = check, byType = Transpose)
# %%


check = MatAdd(
    Transpose(Transpose(Inverse(A))),
    Transpose(Transpose(Transpose(Inverse(Inverse(MatAdd(
        Transpose(Transpose(Inverse(E))), B, R, C.T
    ))))))
)

testSimplifyAlgo(algo = rippleOut, expr = expr_SLaGA, check = check, byType = Transpose)
# %%


check = MatAdd(
    A.I,
    Transpose(Inverse(Inverse(MatAdd(
        E.I, B, R, C.T
    ))))
)
testSimplifyAlgo(algo = factor, expr = expr_SLaGA, check = check, byType = Transpose)

# %%

# old check
check_grouped = Transpose(MatAdd(
    Transpose(Inverse(A)),
    Inverse(Inverse(E.I + B + R + C.T))
))

# new check, after splitting Transpose over MatAdd
check = MatAdd(
    A.I, 
    Transpose(Inverse(Inverse(MatAdd(
        E.I, B, R, C.T
    ))))
)

testSimplifyAlgo(algo = polarize, expr = expr_SLaGA, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = group, expr = expr_SLmGA, check = check, byType = Transpose)
# %%


check = MatMul(
    A,
    Transpose(Inverse(MatAdd(B, C.T)))
)

testSimplifyAlgo(algo = rippleOut, expr = expr_SLmGA, check = check, byType = Transpose)
# %%


check = MatMul(
    A,
    Transpose(Inverse(MatAdd(B, C.T)))
)
testSimplifyAlgo(algo = factor, expr = expr_SLmGA, check = check, byType = Transpose)

# %%


#check_TOFIX = Transpose(MatMul(
#    Transpose(Inverse(MatAdd(C, B.T))),
#    A.T
#))
#check = MatMul(
#    A,
#    Transpose(Inverse(MatAdd(B, C.T)))
#)

# TODO: the different simplified result is occuring after splitting Transpose over MatAdd fix -- is this difference because of that?
check = MatMul(
    A, 
    Inverse(C + B.T)
)

testSimplifyAlgo(algo = polarize, expr = expr_SLmGA, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = group, expr = expr_SLmGA, check = check, byType = Transpose)
# %%


check = MatMul(
    Transpose(Transpose(Transpose(Transpose(Inverse(Inverse(Inverse(A))))))),
    Transpose(Transpose(Transpose(Inverse(Inverse(MatAdd(
        B , Transpose(Transpose(Inverse(E))) , C.T , Transpose(Inverse(R))
    ))))) )
)

testSimplifyAlgo(algo = rippleOut, expr = expr_SLmGA, check = check, byType = Transpose)
# %%



check = MatMul(
    Inverse(Inverse(Inverse(A))),
    Transpose(Inverse(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    ))))
)
testSimplifyAlgo(algo = factor, expr = expr_SLmGA, check = check, byType = Transpose)
# %%

# NOTE: fixed with countTopTransp inner function inside polarize()
# check_TOFIX
check = MatMul(
    Inverse(Inverse(Inverse(A))),
    Transpose(Inverse(Inverse(MatAdd(
        E.I, B, Transpose(Inverse(R)), C.T
    ))))
)
# TODO the difference in res and check might be from the updated wrapDeep function (factoring over args)
testSimplifyAlgo(algo = polarize, expr = expr_SLmGA, check = check, byType = Transpose)
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
testSimplifyAlgo(algo = group, expr = expr_SLaGM, check = check, byType = Transpose)
# %%


check = expr_SLaGM

testSimplifyAlgo(algo = factor, expr = expr_SLaGM, check = check, byType = Transpose)

testSimplifyAlgo(algo = rippleOut, expr = expr_SLaGM, check = check, byType = Transpose)

testSimplifyAlgo(algo = polarize, expr = expr_SLaGM, check = check, byType = Transpose)
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
testSimplifyAlgo(algo = rippleOut, expr = expr_SLaGM, check = check, byType = Transpose)
# %%


check = MatAdd(
    Inverse(Inverse(Inverse(A))),
    Inverse(Inverse(MatAdd(
        R.I, C, Transpose(Inverse(E)), B.T
    )))
)

testSimplifyAlgo(algo = factor, expr = expr_SLaGM, check = check, byType = Transpose)

testSimplifyAlgo(algo = polarize, expr = expr_SLaGM, check = check, byType = Transpose)

# %% --------------------------------------------------------------



# TEST 4a: SL * GM = single symbol * grouped MatMul

expr_SLmGM = MatMul(A, MatMul(B.T, A, R.I))

# %%


check = expr_SLmGM

testSimplifyAlgo(algo = rippleOut, expr = expr_SLmGM, check = check, byType = Transpose)

testSimplifyAlgo(algo = factor, expr = expr_SLmGM, check = check, byType = Transpose)

testSimplifyAlgo(algo = polarize, expr = expr_SLmGM, check = check, byType = Transpose)
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
testSimplifyAlgo(algo = rippleOut, expr = expr_SLmGM, check = check, byType = Transpose)
# %%


check = MatMul(
    Inverse(Inverse(Inverse(A))),
    Transpose(Inverse(Inverse(MatMul(
        B, E.I, C.T, Transpose(Inverse(R))
    ))))
)

testSimplifyAlgo(algo = factor, expr = expr_SLmGM, check = check, byType = Transpose)
# %%
# NOTE: comparing the initial factor (p, result of polarize) with the factored of wrap-deep (fe) and with polarize of fe --- same thing as the initial factoring (p), so no need in this case to call polarize() again, can just stick with initial factor

#p = polarize(Transpose, expr_SLmGM)
#w = wrapDeep(Transpose, p)
#fe = factor(Transpose, w)
#polarize(Transpose, fe)

# OLD check before the wrap-deep-factor fix: 
#check = MatMul(
#    Inverse(Inverse(Inverse(A))),
#    Transpose(Inverse(Inverse(MatMul(
#        B, E.I, C.T, Transpose(Inverse(R))
#    ))))
#)
check = MatMul(
    Inverse(Inverse(Inverse(A))),
    Inverse(Inverse(MatMul(
        R.I, C, Transpose(Inverse(E)), B.T
    )))
)
testSimplifyAlgo(algo = polarize, expr = expr_SLmGM, check = check, byType = Transpose)
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
testSimplifyAlgo(algo = rippleOut, expr = expr_SAaGA, check = check, byType = Transpose)
# %%


check = MatAdd(
    A.T, B.I, C,
    Transpose(Inverse(Inverse(MatAdd(
        B, C.T, D.I
    ))))
)

testSimplifyAlgo(algo = factor, expr = expr_SAaGA, check = check, byType = Transpose)

testSimplifyAlgo(algo = polarize, expr = expr_SAaGA, check = check, byType = Transpose)
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
testSimplifyAlgo(algo = rippleOut, expr = expr_SAaGA, check = check, byType = Transpose)
# %%


check = MatAdd(
    Inverse(Inverse(Inverse(A))),
    Inverse(Inverse(B)),
    C,
    Transpose(Inverse(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    ))))
)

testSimplifyAlgo(algo = factor, expr = expr_SAaGA, check = check, byType = Transpose)

# %%


check = MatAdd(
    Inverse(Inverse(Inverse(A))),
    Inverse(Inverse(B)),
    C,
    Inverse(Inverse(MatAdd(
        R.I, C, Transpose(Inverse(E)), B.T
    )))
)
testSimplifyAlgo(algo = polarize, expr = expr_SAaGA, check = check, byType = Transpose)
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
testSimplifyAlgo(algo = rippleOut, expr = expr_SAmGA, check = check, byType = Transpose)

testSimplifyAlgo(algo = factor, expr = expr_SAmGA, check = check, byType = Transpose)

testSimplifyAlgo(algo = polarize, expr = expr_SAmGA, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = rippleOut, expr = expr_SAmGA, check = check, byType = Transpose)
# %%



check = MatMul(
    Inverse(Inverse(Inverse(A))),
    Inverse(Inverse(B)),
    C,
    Transpose(Inverse(Inverse(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    ))))
)

testSimplifyAlgo(algo = factor, expr = expr_SAmGA, check = check, byType = Transpose)
# %%


check = MatMul(
    Inverse(Inverse(Inverse(A))),
    Inverse(Inverse(B)),
    C,
    Inverse(Inverse(MatAdd(
        R.I, C, Transpose(Inverse(E)), B.T
    )))
)
testSimplifyAlgo(algo = polarize, expr = expr_SAmGA, check = check, byType = Transpose)
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
testSimplifyAlgo(algo = rippleOut, expr = expr_SAaSM, check = check, byType = Transpose)

testSimplifyAlgo(algo = factor, expr = expr_SAaSM, check = check, byType = Transpose)

testSimplifyAlgo(algo = polarize, expr = expr_SAaSM, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = factor, expr = expr_SAaSM, check = check, byType = Transpose)

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

testSimplifyAlgo(algo = polarize, expr = expr_SAaSM, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 8a: SM + GA = single symbol Mul + group symbol Add

expr_SMaGA = MatAdd(
    MatMul(A.T, B.I, C, D.T),
    Inverse(Transpose(MatAdd(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)
# %%

# OLD check (before wrap-deep-factor fix and mat-add-transpose-group fix)
#check = MatAdd(
#    MatMul(A.T, B.I, C, D.T),
#    Transpose(Inverse(MatAdd(
#        B, E.I, Transpose(Inverse(R)), C.T
#    )))
#)
check = MatAdd(
    MatMul(A.T, B.I, C, D.T),
    Inverse(MatAdd(
        R.I, C, Transpose(Inverse(E)), B.T
    ))
)

testSimplifyAlgo(algo = factor, expr = expr_SMaGA, check = check, byType = Transpose)

testSimplifyAlgo(algo = polarize, expr = expr_SMaGA, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = factor, expr = expr_SMaGA, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = polarize, expr = expr_SMaGA, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = factor, expr = expr_SMmGA, check = check, byType = Transpose)
# %%


check = MatMul(
    MatMul(A.T, B.I, C, D.T),
    Inverse(MatAdd(
        R.I, C, Transpose(Inverse(E)), B.T
    ))
)

testSimplifyAlgo(algo = polarize, expr = expr_SMmGA, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = factor, expr = expr_SMmGA, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = polarize, expr = expr_SMmGA, check = check, byType = Transpose)

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

testSimplifyAlgo(algo = factor, expr = expr_SMaGM, check = check, byType = Transpose)
# %%

# NOTE: required second application of polarize
check_GOOD = MatAdd(
    Inverse(MatMul(C, R.I, Transpose(Inverse(E)), B.T, A)),
    MatMul(A.T, B.I, C, D.T)
)

testSimplifyAlgo(algo = polarize, expr = expr_SMaGM, check = check_GOOD, byType = Transpose)
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

testSimplifyAlgo(algo = factor, expr = expr_SMaGM, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = polarize, expr = expr_SMaGM, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = factor, expr = expr_SMmGM, check = check, byType = Transpose)
# %%


check_GOOD = MatMul(
    MatMul(A.T, B.I, C, D.T),
    Inverse(MatMul(
        C, R.I, Transpose(Inverse(E)), B.T, A
    ))
)

testSimplifyAlgo(algo = polarize, expr = expr_SMmGM, check = check_GOOD, byType = Transpose)


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

testSimplifyAlgo(algo = factor, expr = expr_SMmGM, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = polarize, expr = expr_SMmGM, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 12a: GA + GM = group symbol Add + group symbol Mul

expr_GAaGM = MatAdd(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), C.T
    )))
)
# %%



check = MatAdd(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Transpose(Inverse(MatMul(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

testSimplifyAlgo(algo = factor, expr = expr_GAaGM, check = check, byType = Transpose)
# %%


check = MatAdd(
    Inverse(MatMul(C, R.I, Transpose(Inverse(E)), B.T)),
    Inverse(MatAdd(B.I, C, A.T, D.T))
)

testSimplifyAlgo(algo = polarize, expr = expr_GAaGM, check = check, byType = Transpose)
# %% -------------------------------------------------------------


# TEST 12 b: GA + GM = group symbol Add + group symbol Mul (but just small change from C.T to C.T.T)

expr_GAaGM = MatAdd(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), Transpose(Transpose(C))
    )))
)

check_polarize = MatAdd(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Inverse(MatMul(
        C.T, R.I, Transpose(Inverse(E)), B.T
    ))
)

# NOTE: in this case the top-level-transpose count is the same for both resultGroup an d resultAdd (4) so the group expression is returned (as specified in the polarize algo)
testSimplifyAlgo(algo = polarize, expr = expr_GAaGM, check = check_polarize, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 12c: GA + GM = group symbol Add + group symbol Mul (with more layerings per component)

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

testSimplifyAlgo(algo = factor, expr = expr_GAaGM, check = check, byType = Transpose)
# %%


check = MatAdd(
    Transpose(Inverse(Inverse(Inverse(MatAdd(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ))))),
    Inverse(Inverse(MatMul(
        R.I, C, Transpose(Inverse(E)), B.T
    ))),
    Inverse(Inverse(MatMul(
        B, Transpose(Inverse(A)), R
    )))
)

testSimplifyAlgo(algo = polarize, expr = expr_GAaGM, check = check, byType = Transpose)
# %% --------------------------------------------------------------



# TEST 13a: GA * GM = group symbol Add * group symbol Mul

expr_GAmGM = MatMul(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), C.T
    )))
)
# %%


check = MatMul(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Transpose(Inverse(MatMul(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

testSimplifyAlgo(algo = factor, expr = expr_GAmGM, check = check, byType = Transpose)
# %%


check = MatMul(
    Inverse(MatAdd(A.T, B.I, C, D.T)),
    Inverse(MatMul(
        C, R.I, Transpose(Inverse(E)), B.T
    ))
)

testSimplifyAlgo(algo = polarize, expr = expr_GAmGM, check = check, byType = Transpose)
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
# %%



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

testSimplifyAlgo(algo = factor, expr = expr_GAmGM, check = check, byType = Transpose)
# %%



check = MatMul(
    Transpose(Inverse(Inverse(Inverse(MatAdd(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ))))),
    Inverse(Inverse(MatMul(
        R.I, C, Transpose(Inverse(E)), B.T
    ))),
    Inverse(Inverse(MatMul(
        B, Transpose(Inverse(A)), R
    )))
)

testSimplifyAlgo(algo = polarize, expr = expr_GAmGM, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 14a: GM + GM = group symbol Mul + group symbol Mul

expr_GMaGM = MatAdd(
    Inverse(MatMul(A.T, B.I, C, D.T)),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), C.T
    )))
)
# %%



check = MatAdd(
    Inverse(MatMul(A.T, B.I, C, D.T)),
    Transpose(Inverse(MatMul(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

testSimplifyAlgo(algo = factor, expr = expr_GMaGM, check = check, byType = Transpose)
# %%



check = MatAdd(
    Inverse(MatMul(A.T, B.I, C, D.T)),
    Inverse(MatMul(
        C, R.I, Transpose(Inverse(E)), B.T
    ))
)

testSimplifyAlgo(algo = polarize, expr = expr_GMaGM, check = check, byType = Transpose)
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
# %%


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

testSimplifyAlgo(algo = factor, expr = expr_GMaGM, check = check, byType = Transpose)
# %%


check = MatAdd(
    Transpose(Inverse(Inverse(Inverse(MatMul(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ))))),
    Inverse(Inverse(MatMul(
        R.I, C, Transpose(Inverse(E)), B.T
    ))),
    Inverse(Inverse(MatMul(
        B, Transpose(Inverse(A)), R
    )))
)

testSimplifyAlgo(algo = polarize, expr = expr_GMaGM, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 15a: GM * GM = group symbol Mul * group symbol Mul

expr_GMmGM = MatMul(
    Inverse(MatMul(A.T, B.I, C, D.T)),
    Inverse(Transpose(MatMul(
        B, E.I, Inverse(Transpose(R)), C.T
    )))
)
# %%



check = MatMul(
    Inverse(MatMul(A.T, B.I, C, D.T)),
    Transpose(Inverse(MatMul(
        B, E.I, Transpose(Inverse(R)), C.T
    )))
)

testSimplifyAlgo(algo = factor, expr = expr_GMmGM, check = check, byType = Transpose)
# %%



check = MatMul(
    Inverse(MatMul(A.T, B.I, C, D.T)),
    Inverse(MatMul(
        C, R.I, Transpose(Inverse(E)), B.T
    ))
)

testSimplifyAlgo(algo = polarize, expr = expr_GMmGM, check = check, byType = Transpose)
# %% --------------------------------------------------------------


# TEST 15b: GM * GM = group symbol Mul * group symbol Mul

expr_GMmGM = MatMul(
    Transpose(MatMul(A, B)),
    Transpose(MatMul(R, J))
)
# %%



check = expr_GMmGM

testSimplifyAlgo(algo = factor, expr = expr_GMmGM, check = check, byType = Transpose)
# %%


# NOTE: here we can see effect of the layered replace (nesting effect on  multiplication: RJ(AB) instead of just RJAB  )
check = Transpose(MatMul(R, J, A, B))

testSimplifyAlgo(algo = polarize, expr = expr_GMmGM, check = check, byType = Transpose)

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
# %%



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

testSimplifyAlgo(algo = factor, expr = expr_GMmGM, check = check, byType = Transpose)
# %%



check = MatMul(
    Transpose(Inverse(Inverse(Inverse(MatMul(
        Inverse(Inverse(Inverse(A))),
        Inverse(Inverse(B)),
        C,
        D.I
    ))))),
    Inverse(Inverse(MatMul(
        R.I, C, Transpose(Inverse(E)), B.T
    ))),
    Inverse(Inverse(MatMul(
        B, Transpose(Inverse(A)), R
    )))
)

testSimplifyAlgo(algo = polarize, expr = expr_GMmGM, check = check, byType = Transpose)
# FUN TODO: test polarize transpose then inverse and assert same result from inverse then transpose
# %%

#TODO later: SL_a_GA_2 as innermost arg layered with transpse and then combine with any other kind: SL, SA, SM, GA, GM in between the layers. TODO for every kind of expression

# TODO 2: must take all above tests and organize so each function I made gets all those above cases (then separate the "special" cases per function beneath these "general" cases which are made just by combining types of expressions, as opposed to the "biased"/"special" cases.)


# %% -------------------------------------------------------------




# GENERAL TEST 1: inverse out, transpose in
expr = Inverse(Transpose(C*E*B))

# %%
check = expr

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)
# %%


check = Transpose(Inverse(MatMul(C, E, B)))

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)

# %%

# TODO should I leave the inverse / leave Constructors order as they are? Or is it ok for factor to act as rippleOut?
check = Transpose(Inverse(MatMul(C, E, B)))

testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)


# %%

check = Transpose(Inverse(MatMul(C, E, B)))

testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)

# %% -------------------------------------------------------------


# GENERAL TEST 2: transpose out, inverse in
expr = Transpose(Inverse(C*E*B))
# %%


check = expr

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)

testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)

testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)

# %% -------------------------------------------------------------


# GENERAL TEST 3: individual transposes inside inverse
expr = Inverse(B.T * E.T * C.T)
# %%


check = Inverse(Transpose(C*E*B))

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)
# %%


check = expr

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%

check = expr

testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)
# %%

check = Transpose(Inverse(MatMul(C, E, B)))

testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)
# %% -------------------------------------------------------------


# GENERAL TEST 4: individual inverse inside transpose

expr = Transpose(B.I * E.I * C.I)
# %%

check = expr

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)

testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)

testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)


# %% -------------------------------------------------------------


# GENERAL TEST 5 a: individual symbols

Q = MatrixSymbol("Q", a, b)

# %%

(expr1, check1) = (Q, Q)
(expr2, check2) = (Q.T, Q.T)
(expr3, check3) = (C.I, C.I)
(expr4, check4) = (Inverse(Transpose(C)), Inverse(Transpose(C)))

testSimplifyAlgo(algo = group, expr = expr1, check = check1, byType = Transpose)
testSimplifyAlgo(algo = group, expr = expr2, check = check2, byType = Transpose)
testSimplifyAlgo(algo = group, expr = expr3, check = check3, byType = Transpose)
testSimplifyAlgo(algo = group, expr = expr4, check = check4, byType = Transpose)

# %%


(expr1, check1) = (Q, Q)
(expr2, check2) = (Q.T, Q.T)
(expr3, check3) = (C.I, C.I)
(expr4, check4) = (Inverse(Transpose(C)), Transpose(Inverse(C)))

testSimplifyAlgo(algo = rippleOut, expr = expr1, check = check1, byType = Transpose)
testSimplifyAlgo(algo = rippleOut, expr = expr2, check = check2, byType = Transpose)
testSimplifyAlgo(algo = rippleOut, expr = expr3, check = check3, byType = Transpose)
testSimplifyAlgo(algo = rippleOut, expr = expr4, check = check4, byType = Transpose)

# %%


(expr1, check1) = (Q, Q)
(expr2, check2) = (Q.T, Q.T)
(expr3, check3) = (C.I, C.I)
(expr4, check4) = (Inverse(Transpose(C)), Transpose(Inverse(C)))

testSimplifyAlgo(algo = factor, expr = expr1, check = check1, byType = Transpose)
testSimplifyAlgo(algo = factor, expr = expr2, check = check2, byType = Transpose)
testSimplifyAlgo(algo = factor, expr = expr3, check = check3, byType = Transpose)
testSimplifyAlgo(algo = factor, expr = expr4, check = check4, byType = Transpose)

# %%


(expr1, check1) = (Q, Q)
(expr2, check2) = (Q.T, Q.T)
(expr3, check3) = (C.I, C.I)
(expr4, check4) = (Inverse(Transpose(C)), Transpose(Inverse(C)))

testSimplifyAlgo(algo = polarize, expr = expr1, check = check1, byType = Transpose)
testSimplifyAlgo(algo = polarize, expr = expr2, check = check2, byType = Transpose)
testSimplifyAlgo(algo = polarize, expr = expr3, check = check3, byType = Transpose)
testSimplifyAlgo(algo = polarize, expr = expr4, check = check4, byType = Transpose)
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

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)

# %%


check = Transpose(Transpose(Inverse(Inverse(Inverse(MatMul(
    A,
    Transpose(Transpose(Inverse(Inverse(C)))),
    Transpose(Inverse(R)),
    M
))))))

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)

# %%


check = Inverse(Inverse(Inverse(MatMul(
    A, Inverse(Inverse(C)), Transpose(Inverse(R)), M
))))

testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)
# %%


check = Inverse(Inverse(Inverse(MatMul(
    A, Inverse(Inverse(C)), Transpose(Inverse(R)), M
))))

testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)
# %% -------------------------------------------------------------


# GENERAL TEST 6: grouped products

expr = MatMul( Transpose(A*B), Transpose(R*J) )
# %%

check = Transpose(R*J*A*B)

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)

# %%

check = expr

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%

check = expr

testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)
# %%


check = Transpose(MatMul(R, J, A, B))

testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)
# %% -------------------------------------------------------------


# GENERAL TEST 7: individual transposes littered along as matmul

expr = B.T * A.T * J.T * R.T
# %%


check = Transpose(R*J*A*B)

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)
# %%

check = expr

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%

check = expr

testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)
# %%

check = Transpose(MatMul(R, J, A, B))

testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)
# %% -------------------------------------------------------------


# GENERAL TEST 8: inverses mixed with transpose in expr matmul, but with transposes all as the outer expression
L = Transpose(Inverse(MatMul(B, A, R)))

expr = MatMul(A , Transpose(Inverse(R)), Transpose(Inverse(L)) , K , E.T , B.I )
# %%

check = Transpose(
    MatMul( Transpose(Inverse(B)), E , K.T , Inverse(L) , Inverse(R) , A.T)
)

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)
# %%

check = expr

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%

check = MatMul(
    A, Transpose(Inverse(R)), Inverse(Inverse(MatMul(B, A, R))), K, E.T, B.I
)

testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)
# %%


check = MatMul(
    A, Transpose(Inverse(R)), Inverse(Inverse(MatMul(B, A, R))), K, E.T, B.I
)
testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)
# %% -------------------------------------------------------------


# GENERAL TEST 9: mix of inverses and transposes in expr matmul, but this time with transpose not as outer operation, for at least one symbol case.
L = Inverse(Transpose(MatMul(B, A, R)))

expr = MatMul(A , Transpose(Inverse(R)), Inverse(Transpose(L)) , K , E.T , B.I )
# %%


check = Transpose(
    MatMul( Transpose(Inverse(B)), E , K.T , Transpose(Inverse(Transpose(L))) , R.I , A.T)
)
testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)

# %%


check = MatMul(
    A, Transpose(Inverse(R)), Transpose(Transpose(Inverse(Inverse(MatMul(B, A, R))))), K, E.T, B.I
)

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%


check = MatMul(
    A, Transpose(Inverse(R)), Inverse(Inverse(MatMul(B, A, R))), K, E.T , B.I
)
testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)
# %%


check = MatMul(
    A, Transpose(Inverse(R)), Inverse(Inverse(MatMul(B, A, R))), K, E.T , B.I
)
testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)

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

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)

# %%

check = Transpose(MatAdd(
    MatMul(B.T, E, K.T, Inverse(MatMul(B, A, R)), R, A.T), D, K.T
))

testSimplifyAlgo_GroupCombineAdds(expr = expr, check = check, byType = Transpose)

# %%


check = MatAdd(
    MatMul(A, R.T, Transpose(Inverse(MatMul(B, A, R))), K, E.T, B),
    K, D.T
)

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)
testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)
# %%

# TODO this is the old check, before fixing polarize to split transpose over add
# TODO check if indeed the result is as the polarize() intends
check = Transpose(MatAdd(
    D, MatMul(B.T, E, K.T, Inverse(MatMul(B, A, R)), R, A.T), K.T
))
testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)
# %% -------------------------------------------------------------


# GENERAL TEST 11: digger case, very layered expression (with transposes separated so not expecting the grouptranspose to change them). Has inner arg matmul.

expr = Trace(Transpose(Inverse(Transpose(C*D*E))))
# %%

check = expr

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)

testSimplifyAlgo_GroupCombineAdds(expr = expr, check = check, byType = Transpose)

# %%

check = Trace(Transpose(Transpose(Inverse(MatMul(C, D, E)))))

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%



check = Trace(Inverse(MatMul(C, D, E)))

testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)

testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)

# %% -------------------------------------------------------------



# GENERAL TEST 12: very layered expression, but transposes are next to each other, with inner arg as matmul

expr = Trace(Transpose(Transpose(Inverse(C*D*E))))
# %%

check = expr

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)

testSimplifyAlgo_GroupCombineAdds(expr = expr, check = check, byType = Transpose)

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%

check = Trace(Inverse(MatMul(C, D, E)))

testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)

testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)

# %% -------------------------------------------------------------


# GENERAL TEST 13: very layered expression (digger case) with individual transpose and inverses littered in the inner matmul arg.
expr = Transpose(Inverse(Transpose(C.T * A.I * D.T)))
# %%


check = Transpose(Inverse(Transpose(Transpose(
    MatMul(D , Transpose(Inverse(A)), C )
))))

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)
# %%

check = Transpose(Transpose(Inverse(MatMul(C.T, A.I, D.T))))

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)
# %%


check = Inverse(C.T * A.I * D.T)

testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)
# %%


check = Transpose(Inverse(MatMul(
    D, Transpose(Inverse(A)), C
)))
testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)

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

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)
testSimplifyAlgo_GroupCombineAdds(expr = expr, check = check, byType = Transpose)
# %%

check = expr

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)

testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)
# %%

check = Inverse(MatMul(
    Transpose(Inverse(MatMul(
        B, Inverse(B*A*R), Transpose(Inverse(E)), C
    ))),
    MatMul(R, D, A.T)
))
# NOTE even though this looks like the transpose on the left should come out, instead the polarize() does the right thing here since the alternative is RDA.T with outer transpose on the left and an overall transpose over the entire MatMul, so there are more transposes than in this check here (verified by digger hacks)
testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)

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

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)
# %%

check = Transpose(Inverse(MatMul(
    A, D.T, R.T,
    Transpose(Inverse(MatMul(
        C.T, E.I, Transpose(Inverse(MatMul(B, A, R))), B.T
    )))
)))

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)

testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)
# %%


check = Inverse(MatMul(
    Transpose(Inverse(MatMul(
        B, Inverse(B*A*R), Transpose(Inverse(E)), C
    ))),
    MatMul(R, D, A.T)
))

testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)

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

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)

# %%

check = Transpose(Inverse(MatMul(
    A, D.T, R.T,
    Transpose(Inverse(MatMul(
        C.T, E.I, Transpose(Inverse(B*A*R)), B.T
    )))
)))

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)

testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)
# %%


check = Inverse(MatMul(
    Transpose(Inverse(MatMul(
        B, Inverse(B*A*R), Transpose(Inverse(E)), C
    ))),
    MatMul(R, D, A.T)
))

testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)
# %%


check = Transpose(Inverse(MatMul(
    A, D.T, R.T,
    Transpose(Inverse(MatMul(
        C.T, E.I, Transpose(Inverse(B*A*R)), B.T
    )))
)))

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)
testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)
# %%

check = Inverse(MatMul(
    Transpose(Inverse(MatMul(
        B, Inverse(B*A*R), Transpose(Inverse(E)), C
    ))),
    MatMul(R, D, A.T)
))
testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = group, expr = expr, check = check, byType = Transpose)
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

testSimplifyAlgo_GroupCombineAdds(expr = expr, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = rippleOut, expr = expr, check = check, byType = Transpose)
testSimplifyAlgo(algo = factor, expr = expr, check = check, byType = Transpose)

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

testSimplifyAlgo(algo = polarize, expr = expr, check = check, byType = Transpose)

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

testSimplifyAlgo(algo = group, expr = expr_polarize, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = factor, expr = expr_polarize, check = check, byType = Transpose)
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

testSimplifyAlgo(algo = polarize, expr = expr_polarize, check = check, byType = Transpose)




# %% codecell

### GENERAL TEST 17: general ice-breaker test for power simplifications

# TODO separate these expressions and test them separately 

e =  Transpose(MatMul(Transpose(R*J*A.T*B), A*X))
ein = MatAdd(E, Transpose(MatMul(Transpose(R*J*A.T*B), MatPow(X*A, 4))))
eout = MatAdd(E, Transpose(MatMul(
    Transpose(R*J*A.T*B), 
    MatPow(Transpose(J + X*A), 4)
)))
p = Transpose(MatPow(Inverse(MatPow(MatPow(X, 2), 5)), 3))
pmany = MatPow(MatPow(MatPow(MatPow(MatPow(MatPow(MatPow(MatPow(MatPow(MatPow(MatPow(X, -3), 2), 4), 3), -8), 5), -1), -2), 4), 1), -3)

showGroup([e, ein, eout, p, pmany]) 

# %%
# TODO STAR left off here must fix recursion error with isEqMatPow <===> isEq
rippleOut(MatPow, p)





# %%


#### TEST 17 a: testing factor against the pow-related examples
# e
assert factor(Transpose, e) == e
assert factor(MatPow, e) == e

# eout
check = MatAdd(
    E, 
    Transpose(MatMul(
        Transpose(R * J * A.T * B), 
        Transpose(MatPow(J + X*A, 4))
    ))
)
assert factor(Transpose, eout) == check

assert factor(MatPow, eout) == eout

# p
check = MatPow(MatPow(MatPow(Transpose(Inverse(X)), 2), 5), 3)
assert factor(MatPow, p) == check

check = Inverse(Transpose(MatPow(MatPow(MatPow(X, 2), 5), 3)))
assert factor(Inverse, p) == check

assert factor(Transpose, p) == p


# pmany
assert factor(Transpose, pmany) == pmany
assert factor(Inverse, pmany) == pmany

check = MatPow(MatPow(MatPow(MatPow(MatPow(X, 4), -8), 5), 4), -3)
assert factor(MatPow, pmany) == check





# %%
# TODO error PowHolder obj not callable (in applyTypesToExpr() function)
polarize(Transpose, eout)
# %%
polarize(Transpose, ein)
# %%
check = MatAdd(E, Transpose(MatMul(
    Transpose(R*J*A.T*B), 
    Transpose(MatPow(J + X*A, 4))
)))
testSimplifyAlgo(algo = factor, expr = eout, check = check, byType = Transpose)
# %%


check = MatAdd(
    E, 
    Transpose(MatMul(
        Transpose(R*J*A.T*B), 
        Transpose(MatPow(J + X*A, 4))
    ))
)

testSimplifyAlgo(algo = rippleOut, expr = eout, check = check, byType = Transpose)
# %% codecell



check = MatAdd(
    E, 
    MatMul(
        MatPow(J + X*A, 4),
        R*J*A.T * B
    )
)

testSimplifyAlgo(algo = polarize, expr = eout, check = check, byType = Transpose)
# %% codecell
wrapDeep(Transpose, eout)




# %%

### EQUALITY TESTS


otherTypes = list( set(ALL_TYPES_LIST).symmetric_difference(CONSTR_LIST) - set(OP_POW_LIST))


# Blatant example: 
assert not isEq(One, MatPow)

assert all(map(lambda nonPowConstr : not isEq(nonPowConstr, MatPow), otherTypes))

assert all(map(lambda nonPowConstr : not isEq(nonPowConstr, PowHolder), otherTypes))

assert all(map(lambda nonPowConstr : not isEq(nonPowConstr, Pow), otherTypes))



# M-M combo
assert isEq(MatPow, MatPow)
# M-P combo
assert isEq(MatPow, PowHolder)
# P-P combo
assert isEq(PowHolder, Pow)

assert not isEq(Pow, MatPow)


mps = [PowHolder(expo = 1), PowHolder(expo = 2), PowHolder(expo = 3)]
# M-m combo
assert all(map(lambda mp: isEq(MatPow, mp), mps))
# P-m combo
assert all(map(lambda mp: isEq(PowHolder, mp), mps))
# P-m combo
assert all(map(lambda mp: isEq(Pow, mp), mps))

# m-m combo
assert isEq(PowHolder(expo = 1), PowHolder(expo = 1))
assert not isEq(PowHolder(expo = 1), PowHolder(expo = 1123))


# %% codecell
