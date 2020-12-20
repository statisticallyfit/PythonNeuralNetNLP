from sympy.printing.latex import LatexPrinter

from sympy import (Symbol, MatrixSymbol, Matrix, ZeroMatrix, Identity, Add, Mul, MatAdd, MatMul, Determinant, Inverse, Trace, Transpose, Function, derive_by_array, Lambda, Derivative, symbols, diff, sympify)
from sympy.core import Expr, Basic # Basic is base class for all sympy objects
from sympy.matrices.expressions import MatrixExpr
from sympy.abc import x, i, j, a, b
from sympy.core.function import UndefinedFunction, Application

from src.MatrixCalculusStudy.MatrixDerivLib.symbols import Deriv# , RealValuedMatrixFunc


from functools import reduce
from operator import add

# TODO continue using foldl: https://hyp.is/b9hFGg8lEeutuBP4rylAmQ/www.burgaud.com/foldl-foldr-python

# TODO: for this to work with fold, need to have the left ACC arg1 to be a string, so need to first split the ENTIRE expression by MatMul and Pow and Mul until all are flattened and then apply the parentheses around each deriv operation that has a function application and around each RVFM that is an Add.

    # TODO two major rules only for simplicity:
    # RULE 1: if listified expr contains RVFM (Add(_)) then need to wrap that Add expr inside parentheses.
    # RULE 2: same for Pow
    # RULE 3: same for Deriv(f(_), M)
'''    
def twomul(arg1, arg2):
    RV = RealValuedMatrixFunc

    #test_RA_RA = isinstance(arg1, RV) and isinstance(arg1.fExpr, Add) and isinstance(arg2, RV) and isinstance(arg2.fExpr)

    if isinstance(arg1, RealValuedMatrixFunc):
        
        test_RA_RA = isinstance(arg1.fExpr, Add) and isinstance(arg2.fExpr, Add)
        
        test_RA_RM = (isinstance(arg1.fExpr, Add) and isinstance(arg2.fExpr, Mul)) 
        test_RM_RA = (isinstance(arg1.fExpr, Mul) and isinstance(arg2.fExpr, Add))

        test_RA_D = isinstance(arg1.fExpr, Add) and isinstance(arg2, Deriv)

        #test_RM_D no brackets on deriv
        


        if test_RA_RA: 
            return '({}) \\cdot ({})'.format(ppp._print(arg1.fExpr), ppp._print(arg2.fExpr) )

        elif test_RA_RM: 
            return '({}) \\cdot {}'.format(ppp._print(arg1.fExpr), ppp._print(arg2.fExpr))
        elif test_RM_RA: 
            return '{} \\cdot ({})'.format(ppp._print(arg1.fExpr), ppp._print(arg2.fExpr))

        elif test_RA_D: 
            return '({}) \\cdot {}'.format(ppp._print(arg1.fExpr), ppp._print(arg2))
        # ----------------------------------

                

            #elif isinstance(arg2, Deriv) and (isinstance(arg2.fExpr, MatrixSymbol) or Matrix(arg2.byVar).has(arg2.fExpr) ): # if matrix symbol or matrix symbol element, then : 
            #    de = arg2.fExpr
            #    return '({}) \\cdot {}'.format(ppp._print(ra), ppp._print(de))
        
        elif isinstance(arg1.fExpr, Mul):
            rm = arg1.fExpr

            if isinstance(arg2.fExpr, Deriv):
                de = arg2.fExpr
                return '{} \\cdot ({})'.format(ppp._print(rm), ppp._print(de))

    elif isinstance(arg1, Deriv) and isinstance(arg2, Deriv):

        if isinstance(arg1.fExpr, Application) and isinstance(arg2.fExpr, Application):
            return '({}) \\cdot ({})'.format(ppp._print(arg1.fExpr), ppp._print(arg2.fExpr))
        elif isinstance(arg1.fExpr, Application):
            return '({}) \\cdot {}'.format(ppp._print(arg1.fExpr), ppp._print(arg2.fExpr))
        elif isinstance(arg2.fExpr, Application):
            return '{} \\cdot ({})'.format(MyLatexPrinter()._print(arg1), MyLatexPrinter()._print(arg2))
        else:
            return '{} \\cdot {}'.format(MyLatexPrinter()._print(arg1), MyLatexPrinter()._print(arg2))
'''


class MyLatexPrinter(LatexPrinter):

    # GOAL: if the RVMF(Mul) or RVMF(Add) is multiplied by the Deriv obj (then it is MatMul) then include parentheses around the RVMF(Mul) or RVMF(Add) and Deriv obj too
    #def _print_MatMul(self, matmul: MatMul):
    #    return twoMatMul()
    def _print_Deriv(self, deriv: Deriv):

        return '\\frac {\\partial ' + self._print(deriv.dExpr) + '} {\\partial ' + self._print(deriv.byVar) + '}'
        # NOTE: MatrixSymbol has args (symbolletter, sizeRow, sizeCol) so we need args[0] to get its symbol letter
        '''if (isinstance(deriv.dExpr, MatrixSymbol)):
            mat = deriv.dExpr

            return '\\frac {\\partial ' + self._print(mat) + '} {\\partial ' + self._print(deriv.byVar) + '}'

        elif (isinstance(deriv.dExpr, Application)): 
            func_i = deriv.dExpr

            return '\\frac {\\partial } {\\partial ' + self._print(deriv.byVar) + '} ' + self._print(func_i) #+ ' )'

        elif (deriv.dExpr in Matrix(deriv.byVar)):
            elem = deriv.dExpr 

            return '\\frac {\\partial ' + self._print(elem) + '} {\\partial ' + self._print(deriv.byVar) + '}
        '''

'''
    def _print_RealValuedMatrixFunc(self, func_i: RealValuedMatrixFunc):

        return self._print(func_i.fExpr)

    def _print_MatMul(self, matmul: MatMul):
        # printing two at once: split by args and then pass to the twomul function and fold (order will be accounted for so  for instance additions are always printed to left of deriv expressions)
'''
myLatexPrinter = MyLatexPrinter()

