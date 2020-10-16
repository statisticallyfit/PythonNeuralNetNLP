from sympy.printing.latex import LatexPrinter

from sympy import (Symbol, MatrixSymbol, Matrix, ZeroMatrix, Identity, Add, Mul, MatAdd, MatMul, Determinant, Inverse, Trace, Transpose, Function, derive_by_array, Lambda, Derivative, symbols, diff, sympify)
from sympy.core import Expr, Basic # Basic is base class for all sympy objects
from sympy.matrices.expressions import MatrixExpr
from sympy.abc import x, i, j, a, b
from sympy.core.function import UndefinedFunction, Application

from src.MatrixCalculusStudy.LIBSymbolicMatDiff.symbols import Deriv, RealValuedMatrixFunc


from functools import reduce 
from operator import add 

# TODO continue using foldl: https://hyp.is/b9hFGg8lEeutuBP4rylAmQ/www.burgaud.com/foldl-foldr-python

def twomul(arg1, arg2):
    RV = RealValuedMatrixFunc

    #test_RA_RA = isinstance(arg1, RV) and isinstance(arg1.expr, Add) and isinstance(arg2, RV) and isinstance(arg2.expr)

    if isinstance(arg1, RealValuedMatrixFunc):
        
        if isinstance(arg1.expr, Add):
            ra = arg1.expr

            if isinstance(arg2.expr, Add):
                return '({}) \\cdot ({})'.format(ppp._print(ra), ppp._print(arg2.expr) )

            elif isinstance(arg2.expr, Mul):
                rm = arg2.expr
                return '({}) \\cdot {}'.format(ppp._print(ra), ppp._print(rm))

            elif isinstance(arg2, Deriv):# and isinstance(arg2.expr, Application):
                da = arg2.expr
                return '({}) \\cdot ({})'.format(ppp._print(ra), ppp._print(da))

            #elif isinstance(arg2, Deriv) and (isinstance(arg2.expr, MatrixSymbol) or Matrix(arg2.byVar).has(arg2.expr) ): # if matrix symbol or matrix symbol element, then : 
            #    de = arg2.expr
            #    return '({}) \\cdot {}'.format(ppp._print(ra), ppp._print(de))
        
        elif isinstance(arg1.expr, Mul):
            rm = arg1.expr

            if isinstance(arg2.expr, Deriv):
                de = arg2.expr
                return '{} \\cdot ({})'.format(ppp._print(rm), ppp._print(de))

    elif isinstance(arg1, Deriv) and isinstance(arg2, Deriv):

        if isinstance(arg1.expr, Application) and isinstance(arg2.expr, Application):
            return '({}) \\cdot ({})'.format(ppp._print(arg1.expr), ppp._print(arg2.expr))
        elif isinstance(arg1.expr, Application):
            return '({}) \\cdot {}'.format(ppp._print(arg1.expr), ppp._print(arg2.expr))
        elif isinstance(arg2.expr, Application):
            return '{} \\cdot ({})'.format(MyLatexPrinter()._print(arg1), MyLatexPrinter()._print(arg2))
        else:
            return '{} \\cdot {}'.format(MyLatexPrinter()._print(arg1), MyLatexPrinter()._print(arg2))



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

    def _print_MatMul(self, matmul: MatMul):
        # printing two at once: split by args and then pass to the twomul function and fold (order will be accounted for so  for instance additions are always printed to left of deriv expressions)

ppp = MyLatexPrinter()

