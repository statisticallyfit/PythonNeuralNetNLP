# Source of code for this library: https://github.com/mshvartsman/symbolic-mat-diff/blob/master/symbdiff/diff.py


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
from src.MatrixCalculusStudy.MatrixDerivLib.symbols import d, Kron, SymmetricMatrixSymbol, Deriv#, RealValuedMatrixFunc # my deriv here
from src.MatrixCalculusStudy.MatrixDerivLib.simplifications import simplify_matdiff
from src.utils.GeneralUtil import *
from src.MatrixCalculusStudy.MatrixDerivLib.printingLatex import myLatexPrinter

from IPython.display import display, Math
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax', latex_printer= lambda e, **kw: myLatexPrinter.doprint(e))

### PREVIOUS WAY using default sympy latex printer
#from IPython.display import display
#from sympy.interactive import printing
#printing.init_printing(use_latex='mathjax')


# NOTE verified the tensor product rule B x A^T from Alexander Graham book example 5.1: 
# Matrix(derive_by_array(y, x)) == Matrix(TensorProduct(W, eye(3,3)))
# But it seems this needs to be unvectorized to get result W^T (as in add every third row and then add every third column to get W^T. Question here: how does this folding process relate to W^T in the dl_dW_abstract formula from derivtion.py? )

MATRIX_DIFF_RULES = {
    # e = expression, s = SINGLE symbol  respsect to which
    # we want to differentiate. ASSUME S is a MatrixSymbol

    # NOTE: if symbol e is an element of the matrixsymbol s then can do derivative else is just zero.
    Symbol: lambda e, S: Deriv(e, S) if (e in Matrix(S)) else ZeroMatrix(*S.shape),

    MatrixSymbol: lambda e, S: Deriv(e, S) if (e == S) else ZeroMatrix(*e.shape),

    SymmetricMatrixSymbol: lambda e, S: Deriv(e, S) if (e == S) else ZeroMatrix(*e.shape),

    # TODO: need to change this so that only the below acts for when a MatrixSymbol arg but to carry out the matrix chain rules when there are nested functions with different typed arguments (different return type results)
    Application: lambda appFunc, S:  Deriv(appFunc, S) if (appFunc.has(S)) else ZeroMatrix(*S.shape),

    Add: lambda e, S: Add(*[_applyDiffMat(arg, S) for arg in e.args]),

    Mul: lambda e, S: _applyDiffMat(e.args[0], S) if len(e.args) == 1 else Mul(_applyDiffMat(e.args[0], S), Mul(*e.args[1:])) + Mul(e.args[0], _applyDiffMat(Mul(*e.args[1:]), S)),

    MatAdd: lambda e, S: MatAdd(*[_applyDiffMat(arg, S) for arg in e.args]),

    MatMul: lambda e, S: _applyDiffMat(e.args[0], S) if len(e.args) == 1 else MatMul(_applyDiffMat(e.args[0], S), MatMul(*e.args[1:])) + MatMul(e.args[0], _applyDiffMat(MatMul(*e.args[1:]), S)),

    Kron: lambda e, S: _applyDiffMat(e.args[0], S) if len(e.args) == 1 else Kron(_applyDiffMat(e.args[0], S), Kron(*e.args[1:])) + Kron(e.args[0], _applyDiffMat(Kron(*e.args[1:]), S)),

    Determinant: lambda e, S: MatMul(Determinant(e.args[0]), Trace(e.args[0].I * _applyDiffMat(e.args[0], S))),

    # inverse always has 1 arg, so we index
    Inverse: lambda e, S: -Inverse(e.args[0]) * _applyDiffMat(e.args[0], S) * Inverse(e.args[0]),

    # trace always has 1 arg
    Trace: lambda e, S: Trace(_applyDiffMat(e.args[0], S)),
    # transpose also always has 1 arg, index

    Transpose: lambda e, S: Transpose(_applyDiffMat(e.args[0], S))
}



def _applyDiffMat(expression, byVar: MatrixSymbol):
    # The ordinary condition:
    testClass = expression.__class__ in list(MATRIX_DIFF_RULES.keys())
    # Extra condition imposed by me to check that Application type objects like f(A, B, R) get checked in. Otherwise f(A, B, R).__clas__ just gives 'f' and that is not in the keys but using 'isinstance' we can check Application type.
    testInstance = any(map(lambda rulesType: isinstance(expression, rulesType), list(MATRIX_DIFF_RULES.keys())))

    #if expression.__class__ in list(MATRIX_DIFF_RULES.keys()):
    if testClass:
        return MATRIX_DIFF_RULES[expression.__class__](expression, byVar)
    elif testInstance:
        # Find the type of expression using the key (otherwise what is easier way?)
        exprType = list(filter(lambda keyType: isinstance(expression, keyType), MATRIX_DIFF_RULES.keys()))[0]

        return MATRIX_DIFF_RULES[exprType](expression, byVar)

    elif expression.is_constant():
        return 0
    else:
        raise TypeError("Don't know how to differentiate class %s", expression.__class__)


def diffMatrix(expression, byVar: MatrixSymbol):

    def diffAndSimplify(expression, byVar: MatrixSymbol):
        expr = _applyDiffMat(expression, byVar)
        expr = simplify_matdiff(expr, Deriv(byVar, byVar))
        return expr

    return diffAndSimplify(expression, byVar)
    #return [diff_and_simplify(expression, v).doit() for v in variables]


# -----------------------------------


def _diffToGrad(expr, s):
    # if expr is a trace, sum of traces, or scalar times a trace, we can do it
    # scalar times a trace
    if expr.is_Mul and expr.args[0].is_constant() and expr.args[1].is_Trace:
        return (expr.args[0] * expr.args[1].arg.xreplace({d(s): 1}))
    else:
        raise RuntimeError("Don't know how to convert %s to gradient!" % expr)



def gradientMatrix(expr, syms):
    """Compute matrix gradient by matrix differentiation
    """
    diff = diffMatrix(expr, syms)
    grad = [_diffToGrad(e, s) for e, s in zip(diff, syms)]
    return grad

# -----------------------------------



def main():

    n,m,p = 3,3,2

    X = Matrix(n, m, lambda i,j : var_ij('x', i, j))
    W = Matrix(m, p, lambda i,j : var_ij('w', i, j))

    A = MatrixSymbol("A", a, c)
    B = MatrixSymbol("B", c, b)
    R = MatrixSymbol("R", c,c)
    C = MatrixSymbol('C', b, b)
    D = MatrixSymbol('D', b, a)
    L = MatrixSymbol('L', a, c)
    E = MatrixSymbol('E', c, b)

    # Testing with real numbers because the matrix diff function needs real number dimensions 
    # TODO make diffmatrix convert symbolic dims into real dims that match just for the sake of keeping symbolic dims at the end (then replace)
    A_ = MatrixSymbol("A", 4, 3)
    B_ = MatrixSymbol("B", 3, 2)
    R_ = MatrixSymbol("R", 3,3)
    C_ = MatrixSymbol('C', 2, 2)
    D_ = MatrixSymbol('D', 2, 4)
    L_ = MatrixSymbol('L', 4, 3)
    E_ = MatrixSymbol('E', 3, 2)

    # TODO: this doesn't seem correct
    display(diffMatrix(B_ * Inverse(C_) * E_.T, byVar = E_) )

    # TODO this doesn't seem correct --- just puts the diff operator on the A, why doesn't it compare to the diff(X*w) = X^T ???
    display(diffMatrix(B_ * Inverse(C_) * E_.T * L_.T * A_ * E_ * D_ , A_))
    # TODO make the diffmatrix function be able to operate on MatrixSymbol type that has symbols for dimensions, not just real numbers: 
    # B * Inverse(C) * E.T * L.T * A * E * D

    #diffMatrix(A * Inverse(R) * B, R)
    # TODO ERROR
    #assert matDiff_RULES(A*B, A) == diffMatrix(A*B, A)[0].xreplace({d(A) : Deriv(A,A)})

    #display(matDiff_RULES(A*B, A))


    # -----------------------------
    # Bad noncommutative functions have underscore, won't be able to use them in multiplications with Deriv ...
    f_ = Function('f')
    g_ = Function('g')
    h_ = Function('h')
    f = Function('f', commutative=True) # commutative to signify that result is a scalar!
    g = Function('g', commutative=True)
    h = Function('h', commutative=True)

    # Product Rule for Matrices: 
    diffMatrix(f(A,B)*g(A,B), A)


    #RV = RealValuedMatrixFunc
    #ra = RV(f(A) + h(A) + g(A))
    a = f(A) + h(A) + g(A)
    #rm = RV(f(A) * h(A) * g(A))
    m = f(A) * h(A) * g(A)
    df1 = Deriv(f(A), A)
    df2 = Deriv(g(B), B)
    da1 = Deriv(A, A)
    da2 = Deriv(B, B)
    de1 = Deriv(Matrix(A)[0,0], A)
    de2 = Deriv(Matrix(B)[2,1], B)

    # Duplicate entries (for Pow case) so that dimensions match:
    #xs_dup = [ra * ra, rm * rm, df1 * df2, da1 * da2, de1 * de2]
    xs_dup = [a * a, m * m, df1 * df2, da1 * da2, de1 * de2]

    # Trying to create combinations between elements so that dimensions match (sometimes using instead de or da instead of correct combination order to ensure dimensions match)
    xs = [a * m, a * df1, a * da1, a * de1, m * df1, m * da1, m * de1, df1 * de2, da1 * da2, de1 * de2]
    #xs = [ra * rm, ra * df1, ra * da1, ra * de1, rm * df1, rm * da1, rm * de1, df1 * de2, da1 * da2, de1 * de2]

    xs_swap = [m * a, df1 * a, da1 * a, de1 * a, df1 * m, da1 * m, de1 * m, de1 * df2]
    #xs_swap = [rm * ra, df1 * ra, da1 * ra, de1 * ra, df1 * rm, da1 * rm, de1 * rm, de1 * df2]

    #matDiff_RULES(f(A)*g(A), A)
    # Test this if parentheses are printed right:
    # TODO later for now just print the Deriv(f(A), A) on top of the numerator instead of on the side, to avoid having to place parentheses.
    testPrint = de1 * df2 * C * D * de1 * (B + E) * D * A * df2 * E.T
    testPrint_am = de1 * df2 * C * D * de1 * a * R * m * df2 * E.T


if __name__ == "__main__":
    main()