from sympy import (Symbol, MatrixSymbol, Matrix, ZeroMatrix, Identity, Add, Mul, MatAdd, MatMul, Determinant, Inverse, Trace, Transpose, Function, derive_by_array, Lambda, Derivative, symbols, diff)

from sympy.matrices.expressions import MatrixExpr

# NOTE: Application is an applied undefined function like f(x,y) while UndefinedFunction would be just f
from sympy.core.function import UndefinedFunction, Application
from sympy.core import Basic #base class for all sympy objects

from sympy.abc import x, i, j, a, b

from sympy.printing.latex import LatexPrinter, print_latex

# NOTE: Application is an applied undefined function like f(x,y) while UndefinedFunction would be just f
from sympy.core.function import UndefinedFunction, Application
from sympy.core import Basic #base class for all sympy objects





from IPython.display import display, Math
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax')


from typing import *



class d(MatrixExpr):
    """Unevaluated matrix differential (e.g. dX, where X is a matrix)
    """

    def __new__(cls, mat):
        mat = sympify(mat)

        if not mat.is_Matrix:
            raise TypeError("input to matrix derivative, %s, is not a matrix" % str(mat))

        return Basic.__new__(cls, mat)

    @property
    def arg(self):
        return self.args[0]

    @property
    def shape(self):
        return (self.arg.rows, self.arg.cols)



# %%


class Deriv(MatrixExpr):
    def __init__(self, expr: Basic, byVar: MatrixSymbol):
        self.expr = expr
        self.byVar = byVar 

    def __new__(cls, expr: Basic, byVar: MatrixSymbol):
        return Basic.__new__(cls, expr, byVar)



class MyLatexPrinter(LatexPrinter):

    def _print_Deriv(self, deriv: Deriv):

        # NOTE: MatrixSymbol has args (symbolletter, sizeRow, sizeCol) so we need args[0] to get its symbol letter
        if len(deriv.expr.args) == 1: 
            return '\\displaystyle \\frac {\\partial ' + self._print(deriv.expr.args[0]) + '} {\\partial ' + self._print(deriv.byVar) + '}'

        else:
            return '\\displaystyle \\frac {\\partial } {\\partial ' + self._print(deriv.byVar) + '} ' + self._print(deriv.expr) #+ ')'


#def print_my_latex(expr):
#    display(Math(MyLatexPrinter().doprint(expr)))

printer = MyLatexPrinter()

from IPython.display import display, Math
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax', latex_printer= lambda e, **kw: printer.doprint(e))



# %%

if __name__ == "__main__":
    A = MatrixSymbol("A", 4, 3)
    B = MatrixSymbol("B", 3, 2)
    R = MatrixSymbol("R", 3,3)

    Deriv(A*B, A)
    print(Deriv(A*B, A))
#    matLatex(A*B, A)
# %%




### NOTE: here the d(M) is made a matrix instance, so that this assertion is true: 
# isinstance(d(M), MatrixExpr)
# where M is a Matrix or MatrixSymbol

class SymmetricMatrixSymbol(MatrixSymbol):
    """Symmetric matrix
    """
    is_Symmetric = True

    def _eval_transpose(self):
        return self

    def _eval_inverse(self):
        inv = Inverse(self)
        inv.is_Symmetric = True
        return inv

Kron = Function("Kron",commutative=False)