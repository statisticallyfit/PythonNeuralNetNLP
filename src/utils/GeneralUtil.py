



from IPython.display import display

from typing import * 

from sympy import Symbol, Function, Matrix, MatrixSymbol, ZeroMatrix

import itertools
from functools import reduce



# Finds indices where a condition is met
def find(xs: List[Any], condition: bool) -> List[int]:
    indices = [i  for i in range(0, len(xs))  if (condition)]

    return indices


# Finds indices where an expression matches elements in the given xs list.
def findWhere(xs: List[Any], expr: Any) -> List[int]:
    indices = [i  for i in range(0, len(xs))  if (xs[i] == expr)]

    return indices




# For showing / displaying items as group all at once in code cells in sympy latex
def showGroup(group: List[Any]) -> None:
    list(map(lambda elem : display(elem), group))

    return None




# For creating symbols from sympy
def var_i(letter: str, i: int) -> Symbol:
    letter_i = Symbol('{}_{}'.format(letter, i), is_commutative=True)
    return letter_i


def var_ij(letter: str, i: int, j: int) -> Symbol:
    letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
    return letter_ij


def func_i(fLetter: str, i: int, xLetter: str, xLen: int):
    xs = [var_i(xLetter, i+1) for i in range(xLen)]
    func_i = Function('{}_{}'.format(fLetter, i + 1), is_commutative=True)(*xs)
    return func_i

def func_ij(fLetter: str, i: int, j: int, X: Matrix):
    #xs = [var_i(xLetter, i+1) for i in range(xLen)]
    func_ij = Function('{}_{}{}'.format(fLetter, i + 1, j + 1))(*X)
    return func_ij


# Vec operator: stacks columns of matrices
import itertools

#vec = lambda M : list(itertools.chain(*M.T.tolist()))
def vec(matrix: Matrix) -> List[Symbol]:
    return list(itertools.chain(*matrix.T.tolist()))




def elemMat(shape: Tuple[int, int], i: int, j: int) -> Matrix:
    Z: Matrix = Matrix(ZeroMatrix(*shape))
    Z[i, j] = 1
    return Z 



def composeTwoFunctions(f, g):
    return lambda *a, **kw: f(g(*a, **kw))

def compose(*fs):
    return reduce(composeTwoFunctions, fs)

