from sympy import (Symbol, MatrixSymbol, Matrix, MatrixExpr, ZeroMatrix, Identity, Add, Mul, MatAdd, MatMul, Determinant, Inverse, Trace, Transpose, Function, derive_by_array, Lambda, Derivative, symbols, diff, sympify)

#from sympy.matrices.expressions import MatrixExpr

# NOTE: Application is an applied undefined function like f(x,y) while UndefinedFunction would be just f
from sympy.core.function import UndefinedFunction, Application
from sympy.core import Basic #base class for all sympy objects

from sympy.abc import x, i, j, a, b

# NOTE: Application is an applied undefined function like f(x,y) while UndefinedFunction would be just f
from sympy.core.function import UndefinedFunction, Application
from sympy.core import Basic #base class for all sympy objects


from typing import *

#  Setting the paths for importing:
import sys
import os

PATH: str = '/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP'

MATDIFF_PATH: str = PATH + "/src/MatrixCalculusStudy/LIBSymbolicMatDiff/"

UTIL_PATH: str = PATH + "/src/utils/"

sys.path.append(PATH)
sys.path.append(MATDIFF_PATH)
sys.path.append(UTIL_PATH)

# Importing the custom files:

### File imports
#from .symbols import d, Kron, SymmetricMatrixSymbol
#from .simplifications import simplify_matdiff

### Interactive imports:
#from symbols import d, Kron, SymmetricMatrixSymbol
#from simplifications import simplify_matdiff
# NOTE: need these imports below when executing in Python Interactive. Here these imports don't really work for the file itself, only in interactive.
from src.MatrixCalculusStudy.LIBSymbolicMatDiff.symbols import d, Kron, SymmetricMatrixSymbol, Deriv, RealValuedMatrixFunc # my deriv here
from src.MatrixCalculusStudy.LIBSymbolicMatDiff.simplifications import simplify_matdiff



from src.utils.GeneralUtil import *


# Import the latex printer (for Deriv)
from src.MatrixCalculusStudy.LIBSymbolicMatDiff.printingLatex import myLatexPrinter

from IPython.display import display, Math
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax', latex_printer= lambda e, **kw: myLatexPrinter.doprint(e))

### PREVIOUS WAY using default sympy latex printer
#from IPython.display import display
#from sympy.interactive import printing
#printing.init_printing(use_latex='mathjax')



MY_RULES = {
    # e = expression, s = SINGLE symbol  respsect to which
    # we want to differentiate. ASSUME S is a MatrixSymbol

    # NOTE: if symbol e is an element of the matrixsymbol s then can do derivative else is just zero.
    Symbol: lambda e, S: Deriv(e, S) if (e in Matrix(S)) else ZeroMatrix(*S.shape),

    MatrixSymbol: lambda e, S: Deriv(e, S) if (e == S) else ZeroMatrix(*e.shape),

    SymmetricMatrixSymbol: lambda e, S: Deriv(e, S) if (e == S) else ZeroMatrix(*e.shape),
    #Symbol: lambda e, s: d(e) if (e in s) else 0,
    #MatrixSymbol: lambda e, s: d(e) if (e in s) else ZeroMatrix(*e.shape),
    #SymmetricMatrixSymbol: lambda e, s: d(e) if (e in s) else ZeroMatrix(*e.shape),

    Application: lambda appFunc, S: _matDiff_apply_RULES(RealValuedMatrixFunc(appFunc), S),
    #lambda appFunc, S: Deriv(appFunc, S) if (appFunc.has(S)) else ZeroMatrix(*S.shape),
    #Deriv(appFunc, S) if (appFunc.has(S)) else ZeroMatrix(*S.shape),

    RealValuedMatrixFunc: lambda rvmf, S: Deriv(rvmf.fa, S) if (rvmf.fa.has(S)) else ZeroMatrix(*S.shape),

    Add: lambda e, S: Add(*[_matDiff_apply_RULES(arg, S) for arg in e.args]),

    Mul: lambda e, S: _matDiff_apply_RULES(e.args[0], S) if len(e.args)==1 else Mul(_matDiff_apply_RULES(e.args[0],S),Mul(*e.args[1:])) + Mul(e.args[0], _matDiff_apply_RULES(Mul(*e.args[1:]),S)),

    MatAdd: lambda e, S: MatAdd(*[_matDiff_apply_RULES(arg, S) for arg in e.args]),

    MatMul: lambda e, S: _matDiff_apply_RULES(e.args[0], S) if len(e.args)== 1 else MatMul(_matDiff_apply_RULES(e.args[0],S),MatMul(*e.args[1:])) + MatMul(e.args[0], _matDiff_apply_RULES(MatMul(*e.args[1:]),S)),

    Kron: lambda e, S: _matDiff_apply_RULES(e.args[0],S) if len(e.args)==1 else Kron(_matDiff_apply_RULES(e.args[0],S),Kron(*e.args[1:])) + Kron(e.args[0],_matDiff_apply_RULES(Kron(*e.args[1:]),S)),

    Determinant: lambda e, S: MatMul(Determinant(e.args[0]), Trace(e.args[0].I*_matDiff_apply_RULES(e.args[0], S))),

    # inverse always has 1 arg, so we index
    Inverse: lambda e, S: -Inverse(e.args[0]) * _matDiff_apply_RULES(e.args[0], S) * Inverse(e.args[0]),

    # trace always has 1 arg
    Trace: lambda e, S: Trace(_matDiff_apply_RULES(e.args[0], S)),
    # transpose also always has 1 arg, index

    Transpose: lambda e, S: Transpose(_matDiff_apply_RULES(e.args[0], S))
}


# THe matrix diff rules:
MATRIX_DIFF_RULES = {
    # e =expression, s = a list of symbols respsect to which
    # we want to differentiate
    Symbol: lambda e, s: d(e) if (e in s) else 0,
    MatrixSymbol: lambda e, s: d(e) if (e in s) else ZeroMatrix(*e.shape),
    SymmetricMatrixSymbol: lambda e, s: d(e) if (e in s) else ZeroMatrix(*e.shape),
    #Symbol: lambda e, s: d(e) if (e in s) else 0,
    #MatrixSymbol: lambda e, s: d(e) if (e in s) else ZeroMatrix(*e.shape),
    #SymmetricMatrixSymbol: lambda e, s: d(e) if (e in s) else ZeroMatrix(*e.shape),
    Add: lambda e, s: Add(*[_matDiff_apply(arg, s) for arg in e.args]),
    Mul: lambda e, s: _matDiff_apply(e.args[0], s) if len(e.args)==1 else Mul(_matDiff_apply(e.args[0],s),Mul(*e.args[1:])) + Mul(e.args[0], _matDiff_apply(Mul(*e.args[1:]),s)),
    MatAdd: lambda e, s: MatAdd(*[_matDiff_apply(arg, s) for arg in e.args]),

    MatMul: lambda e, s: _matDiff_apply(e.args[0], s) if len(e.args)== 1 else MatMul(_matDiff_apply(e.args[0],s),MatMul(*e.args[1:])) + MatMul(e.args[0], _matDiff_apply(MatMul(*e.args[1:]),s)),

    Kron: lambda e, s: _matDiff_apply(e.args[0],s) if len(e.args)==1 else Kron(_matDiff_apply(e.args[0],s),Kron(*e.args[1:]))
                  + Kron(e.args[0],_matDiff_apply(Kron(*e.args[1:]),s)),
    Determinant: lambda e, s: MatMul(Determinant(e.args[0]), Trace(e.args[0].I*_matDiff_apply(e.args[0], s))),
    # inverse always has 1 arg, so we index
    Inverse: lambda e, s: -Inverse(e.args[0]) * _matDiff_apply(e.args[0], s) * Inverse(e.args[0]),
    # trace always has 1 arg
    Trace: lambda e, s: Trace(_matDiff_apply(e.args[0], s)),
    # transpose also always has 1 arg, index
    Transpose: lambda e, s: Transpose(_matDiff_apply(e.args[0], s))
}


def _matDiff_apply_RULES(expression, byVar: MatrixSymbol):
    # The ordinary condition:
    testClass = expression.__class__ in list(MY_RULES.keys())
    # Extra condition imposed by me to check that Application type objects like f(A, B, R) get checked in. Otherwise f(A, B, R).__clas__ just gives 'f' and that is not in the keys but using 'isinstance' we can check Application type.
    testInstance = any(map(lambda rulesType: isinstance(expression, rulesType), list(MY_RULES.keys())))

    #if expression.__class__ in list(MY_RULES.keys()):
    if testClass:
        return MY_RULES[expression.__class__](expression, byVar)
    elif testInstance:
        # Find the type of expression using the key (otherwise what is easier way?)
        exprType = list(filter(lambda keyType: isinstance(expression, keyType), MY_RULES.keys()))[0]

        return MY_RULES[exprType](expression, byVar)

    elif expression.is_constant():
        return 0
    else:
        raise TypeError("Don't know how to differentiate class %s", expression.__class__)


def matDiff_RULES(expression, variable: MatrixSymbol):

    def diff_and_simplify(expression, byVar: MatrixSymbol):
        expr = _matDiff_apply_RULES(expression, byVar)
        expr = simplify_matdiff(expr, Deriv(byVar, byVar))
        return expr

    return diff_and_simplify(expression, variable)
    #return [diff_and_simplify(expression, v).doit() for v in variables]


def _matDiff_apply(expression, byVar):
    if expression.__class__ in list(MATRIX_DIFF_RULES.keys()):
        return MATRIX_DIFF_RULES[expression.__class__](expression, byVar)
    elif expression.is_constant():
        return 0
    else:
        raise TypeError("Don't know how to differentiate class %s", expression.__class__)


def matDiff(expression, variables):
    # diff wrt 1 element wrap in list
    try:
        _ = variables.__iter__
    except AttributeError:
        variables = [variables]

    def diff_and_simplify(expression, byVar: List[Symbol]):
        expr = _matDiff_apply(expression, [byVar])
        expr = simplify_matdiff(expr, d(byVar))
        return expr

    return [diff_and_simplify(expression, v).doit() for v in variables]


# -----------------------------------


def _diff_to_grad(expr, s):
    # if expr is a trace, sum of traces, or scalar times a trace, we can do it
    # scalar times a trace
    if expr.is_Mul and expr.args[0].is_constant() and expr.args[1].is_Trace:
        return (expr.args[0] * expr.args[1].arg.xreplace({d(s): 1}))
    else:
        raise RuntimeError("Don't know how to convert %s to gradient!" % expr)



def matGrad(expr, syms):
    """Compute matrix gradient by matrix differentiation
    """
    diff = matDiff(expr, syms)
    grad = [_diff_to_grad(e, s) for e, s in zip(diff, syms)]
    return grad

# -----------------------------------



def var_i(letter: str, i: int) -> Symbol:
    letter_i = Symbol('{}_{}'.format(letter, i), is_commutative=True)
    return letter_i


def var_ij(letter: str, i: int, j: int) -> Symbol:
    letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
    return letter_ij


def func(fLetter: str, i: int, xLetter, xLen):
    xs = [var_i(xLetter, i+1) for i in range(xLen)]
    func_i = Function('{}_{}'.format(fLetter, i + 1))(*xs)
    return func_i


def main():

    n,m,p = 3,3,2

    X = Matrix(n, m, lambda i,j : var_ij('x', i, j))
    W = Matrix(m, p, lambda i,j : var_ij('w', i, j))

    #A = MatrixSymbol('X',n,m)
    #B = MatrixSymbol('W',m,p)
    A = MatrixSymbol("A", 4, 3)
    B = MatrixSymbol("B", 3, 2)
    R = MatrixSymbol("R", 3,3)

    #matDiff(A * Inverse(R) * B, R)

    assert matDiff_RULES(A*B, A) == matDiff(A*B, A)[0].xreplace({d(A) : Deriv(A,A)})

    display(matDiff_RULES(A*B, A))


    # -----------------------------
    f = Function('f')
    g = Function('g')
    h = Function('h')

    RV = RealValuedMatrixFunc
    ra = RV(f(A) + h(A) + g(A))
    rm = RV(f(A) * h(A) * g(A))
    df1 = Deriv(f(A), A)
    df2 = Deriv(g(B), B)
    da1 = Deriv(A, A)
    da2 = Deriv(B, B)
    de1 = Deriv(Matrix(A)[0,0], A)
    de2 = Deriv(Matrix(B)[2,1], B)

    # Duplicate entries (for Pow case) so that dimensions match: 
    xs_dup = [ra * ra, rm * rm, df1 * df2, da1 * da2, de1 * de2]

    # Trying to create combinations between elements so that dimensions match (sometimes using instead de or da instead of correct combination order to ensure dimensions match)
    xs = [ra * rm, ra * df1, ra * da1, ra * de1, rm * df1, rm * da1, rm * de1, df1 * de2, da1 * da2, de1 * de2]

    xs_swap = [rm * ra, df1 * ra, da1 * ra, de1 * ra, df1 * rm, da1 * rm, de1 * rm, de1 * df2]

    #matDiff_RULES(f(A)*g(A), A)


if __name__ == "__main__":
    main()