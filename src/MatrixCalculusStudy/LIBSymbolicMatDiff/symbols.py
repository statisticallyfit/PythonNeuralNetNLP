from sympy import (Symbol, MatrixSymbol, Matrix, ZeroMatrix, Identity, Add, Mul, MatAdd, MatMul, Determinant, Inverse, Trace, Transpose, Function, derive_by_array, Lambda, Derivative, symbols, diff, sympify)

from sympy.core import Expr, Basic # Basic is base class for all sympy objects
from sympy.matrices.expressions import MatrixExpr
#from sympy.matrices.expressions import MatrixExpr

# NOTE: Application is an applied undefined function like f(x,y) while UndefinedFunction would be just f
from sympy.core.function import UndefinedFunction, Application


from sympy.abc import x, i, j, a, b

# NOTE: Application is an applied undefined function like f(x,y) while UndefinedFunction would be just f
from sympy.core.function import UndefinedFunction, Application
from sympy.core import Basic #base class for all sympy objects


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


# GOal: make functions multiplyable with matrix symbols
# NOTE: need this to extend Expr not MatrixExpr so can multiply by ANY MatrixSymbol. Can now do func * Deriv(A, B) where func = RealValuedMatrixFunc(f(A,B,R))
class RealValuedMatrixFunc(Expr):
    def __init__(self, func: Application):
        self.fa = func 
        self.f = func.__class__ #get function letter name
        self.variables = func.args 

    def __new__(cls, func: Application):
        func = sympify(func)

        if not isinstance(func, Application):
            raise TypeError("must pass in an applied UndefinedFunction")

        return Basic.__new__(cls, func)

    # this shape doesn't work when want to multiply by actual shaped MatrixSymbols. 
    #@property 
    #def shape(self):
    #    return (1, 1) # assuming real-valued function that takes matrix argument


# My class that stands in place of above d() to represent derivative as fraction for only function or matrixsymbol arguments:
class Deriv(MatrixExpr):
    def __init__(self, expr, byVar: MatrixSymbol):
        self.expr = expr
        self.byVar = byVar

    def __new__(cls, expr, byVar: MatrixSymbol):
        expr = sympify(expr)
        byVar = sympify(byVar)

        # If numerator is not a matrixsymbol or applied function, throw error

        if not isinstance(byVar, MatrixSymbol):
            raise TypeError("variable with respect to which we differentiate must be MatrixSymbol")


        # NOTE: assuming the applied function takes in a matrix symbol only or multiple matrix symbols or an element in the byVar MatrixSymbol

        if not isinstance(expr, MatrixSymbol) and not isinstance(expr, Application) and not (expr in Matrix(byVar)):
            raise TypeError("input to matrix derivative, %s, is not a MatrixSymbol or Application (applied UndefinedFunction) or an element in byVar" % str(expr))

        elif isinstance(expr, Application):
            func = expr 

            # TODO: what if functino has ssclar argument and that scalar is an element of another argumnt matrix?
            if not func.has(byVar):
                raise AttributeError("Applied function must contain the argument by which we differentiate")

            # If not all arguments are matrix type throw error    
            elif not all(map(lambda theArg: isinstance(theArg, MatrixSymbol), func.args)):
                raise TypeError("function must have MatrixSymbol argument only")

        

        return Basic.__new__(cls, expr, byVar)


    @property
    def shape(self):
        if isinstance(self.expr, Application):
            return (self.byVar.rows, self.byVar.cols)

        # This else case satisfies both cases of when expr is MatrixSymbol and when it is matrix element. 
        else: #if isinstance(self.expr, MatrixSymbol):
            mat = self.expr

            if(mat != self.byVar):
                return (mat.rows * self.byVar.rows, mat.cols * self.byVar.cols)

            # NOTE wrong implementation here below, just copying d(A) implementation to satisfy the matrix multiplication of Deriv(A, A) * B
            return (self.byVar.rows, self.byVar.cols)

# NOTE: copying just d(A)'s shape implementation even if not accurate for dA / dA shape because need to get multiplication working. Need to be able to write Deriv(A, A) * B but my implementation below won't l et me due to squaring of the shapes ...
'''        
        # TODO using the block matrix dimension result here (using result of diff's matrix shape instead of the TensorPRoduct diff shape)
        if isinstance(self.expr, Application):

            # Squaring since we differentiate by the argument inside func, which is the same as the byVar
            return (self.byVar.rows**2, self.byVar.cols**2)

        elif isinstance(self.expr, MatrixSymbol):
            mat = self.expr
     
            return (mat.rows * self.byVar.rows, mat.cols * self.byVar.cols)

        elif self.expr in Matrix(byVar):
            # then expr is a matrix element of byVar matrixsymbol

            return (self.byVar.rows, self.byVar.cols)
'''



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