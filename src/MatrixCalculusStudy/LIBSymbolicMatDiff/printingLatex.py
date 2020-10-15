from sympy.printing.latex import LatexPrinter

from sympy import (Symbol, MatrixSymbol, Matrix, ZeroMatrix, Identity, Add, Mul, MatAdd, MatMul, Determinant, Inverse, Trace, Transpose, Function, derive_by_array, Lambda, Derivative, symbols, diff, sympify)
from sympy.core import Expr, Basic # Basic is base class for all sympy objects
from sympy.matrices.expressions import MatrixExpr
from sympy.abc import x, i, j, a, b
from sympy.core.function import UndefinedFunction, Application

from src.MatrixCalculusStudy.LIBSymbolicMatDiff.symbols import Deriv, RealValuedMatrixFunc




# TODO continue using foldl: https://www.burgaud.com/foldl-foldr-python

def twoMatMul(arg1, arg2):

    if isinstance(arg1, RealValuedMatrixFunc):
        
        if isinstance(arg1.expr, Add):
            ra = arg1.expr

            if isinstance(arg2.expr, Add):
                return '({}) \\cdot ({})'.format(ra, arg2.expr)
            elif isinstance(arg2.expr, Mul):
                rm = arg2.expr
                return '({}) \\cdot {}'.format(ra, rm)
            elif isinstance(arg2.expr, Deriv):
                de = arg2.expr
                return '({}) \\cdot ({})'.format(ra, de)
        
        elif isinstance(arg1.expr, Mul):
            rm = arg1.expr

            if isinstance(arg2, expr, Deriv):
                de = arg2.expr
                return '{} \\cdot ({})'.format(rm, de)

    elif isinstance(arg1, Deriv) and isinstance(arg2, Deriv):
        if isinstance(arg1.expr, Application) and isinstance(arg2.expr, Application):
            return '({}) \\cdot ({})'.format(self._print(arg1), self._print(arg2))
        elif isinstance(arg1.expr, Application):
            return '({}) \\cdot {}'.format(self._print(arg1), self._print(arg2))
        elif isinstance(arg2.expr, Application):
            return '{} \\cdot ({})'.format(self._print(arg1), self._print(arg2))
        else:
            return '{} \\cdot {}'.format(self._print(arg1), self._print(arg2))

class MyLatexPrinter(LatexPrinter):

    # GOAL: if the RVMF(Mul) or RVMF(Add) is multiplied by the Deriv obj (then it is MatMul) then include parentheses around the RVMF(Mul) or RVMF(Add) and Deriv obj too
    #def _print_MatMul(self, matmul: MatMul):
    #    return twoMatMul()
    def _print_Deriv(self, deriv: Deriv):

        # NOTE: MatrixSymbol has args (symbolletter, sizeRow, sizeCol) so we need args[0] to get its symbol letter
        if (isinstance(deriv.expr, MatrixSymbol)):
            mat = deriv.expr

            return '\\frac {\\partial ' + self._print(mat) + '} {\\partial ' + self._print(deriv.byVar) + '}'
        elif (isinstance(deriv.expr, Application)): 
            func = deriv.expr

            return '\\frac {\\partial } {\\partial ' + self._print(deriv.byVar) + '} ' + self._print(func) #+ ' )'

        elif (deriv.expr in Matrix(deriv.byVar)):
            elem = deriv.expr 

            return '\\frac {\\partial ' + self._print(elem) + '} {\\partial ' + self._print(deriv.byVar) + '}'


    def _print_RealValuedMatrixFunc(self, func: RealValuedMatrixFunc):

        return self._print(func.expr)


myLatexPrinter = MyLatexPrinter()

