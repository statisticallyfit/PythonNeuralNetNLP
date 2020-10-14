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
from sympy import sympify

class Deriv(MatrixExpr):
    def __init__(self, mat: MatrixSymbol):
        self.mat = mat

    def __new__(cls, mat: MatrixSymbol):
        return Basic.__new__(cls, mat)

    
    @property
    def shape(self):
        # TODO assumpin the expr function is real-valued, then would d(f(X))/dX have shape equal to X? 
        # TODO when expr is just matrix symbol then shape of dX/dX is (rows^2, cols^2) of X
        # To check the below, try derive_by_array(Matrix(mat), Matrix(mat) to see it is that kind of identity matrix. 
        
        #return (self.mat.rows**2, self.mat.cols**2)
        # NOTE: (update) if we put the above then cannot do matmul predictably anymore so just return same shape as the argument
        return (self.mat.rows, self.mat.cols)    



class MyLatexPrinter(LatexPrinter):

    def _print_Deriv(self, deriv: Deriv):

        # NOTE: MatrixSymbol has args (symbolletter, sizeRow, sizeCol) so we need args[0] to get its symbol letter
        
        return '\\displaystyle \\frac {\\partial ' + self._print(deriv.mat) + '} {\\partial ' + self._print(deriv.mat) + '}'


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

    DE(A)
    print(DE(A))
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