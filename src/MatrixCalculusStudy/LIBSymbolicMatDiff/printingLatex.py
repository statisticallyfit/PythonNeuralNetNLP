from sympy.printing.latex import LatexPrinter, print_latex

from sympy import (Symbol, MatrixSymbol, Matrix, ZeroMatrix, Identity, Add, Mul, MatAdd, MatMul, Determinant, Inverse, Trace, Transpose, Function, derive_by_array, Lambda, Derivative, symbols, diff, sympify)
from sympy.core import Expr, Basic # Basic is base class for all sympy objects
from sympy.matrices.expressions import MatrixExpr
from sympy.core.function import UndefinedFunction, Application
from sympy.abc import x, i, j, a, b
from sympy.core.function import UndefinedFunction, Application
from sympy.core import Basic #base class for all sympy objects
from sympy.core.function import UndefinedFunction, Application

from src.MatrixCalculusStudy.LIBSymbolicMatDiff.symbols import Deriv, RealValuedMatrixFunc


class MyLatexPrinter(LatexPrinter):

    def _print_Deriv(self, deriv: Deriv):

        # NOTE: MatrixSymbol has args (symbolletter, sizeRow, sizeCol) so we need args[0] to get its symbol letter
        if (isinstance(deriv.expr, MatrixSymbol)):
            mat = deriv.expr

            return '\\displaystyle \\frac {\\partial ' + self._print(mat) + '} {\\partial ' + self._print(deriv.byVar) + '}'
        elif (isinstance(deriv.expr, Application)): 
            func = deriv.expr

            return '\\displaystyle \\frac {\\partial } {\\partial ' + self._print(deriv.byVar) + '} ' + self._print(func) #+ ' )'

        elif (deriv.expr in Matrix(deriv.byVar)):
            elem = deriv.expr 

            return '\\displaystyle \\frac {\\partial ' + self._print(elem) + '} {\\partial ' + self._print(deriv.byVar) + '}'


    def _print_RealValuedMatrixFunc(self, func: RealValuedMatrixFunc):

        return '\\displaystyle ' + self._print(func.fa)


myLatexPrinter = MyLatexPrinter()

