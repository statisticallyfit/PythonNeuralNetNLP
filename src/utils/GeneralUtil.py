



from IPython.display import display



### Sympy 
from sympy import det, Determinant, Trace, Transpose, Inverse, Function, Lambda, HadamardProduct, Matrix, MatrixExpr, Expr, Symbol, derive_by_array, MatrixSymbol, Identity, ZeroMatrix, Derivative, symbols, diff

from sympy import srepr , simplify

from sympy import tensorcontraction, tensorproduct, preorder_traversal
from sympy.functions.elementary.piecewise import Undefined
from sympy.physics.quantum.tensorproduct import TensorProduct

from sympy.abc import x, i, j, a, b, c

from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matmul import MatMul



### Python tools

from typing import * 
import itertools
from functools import reduce # for folding
import operator # for folding


### Tensors
import numpy as np
from numpy import ndarray

import torch
import torch.tensor as tensor
Tensor = torch.Tensor
LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor



### My functions --------------------------------


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
    letter_i = Symbol('{}_{}'.format(letter, i+1), is_commutative=True)
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







def checkBroadcastable(x: Tensor, y: Tensor) -> bool:
    '''

    '''
    prependOnes = Tensor([1 for i in range(0, abs(x.ndim - y.ndim))])
    (smallestTensor, largestTensor) = (y, x) if y.ndim < x.ndim else (x, y)
    onesSmallestSize = torch.cat((prependOnes, Tensor(smallestTensor.size())), 0)
    pairs = list(zip(Tensor(largestTensor.size()).tolist(), onesSmallestSize.tolist() )) 
    batchDimPairs = pairs[0:-2] # all the dims except the last two are the batch dimension pairs
    isBroadcastable = all(map(lambda p: p[0] == 1 or p[1] == 1 or p[0] == p[1], batchDimPairs))

    return isBroadcastable

### TEsts for checking broadcastable function works correctly: 
#x = torch.randn(8,2,6,7,2,1,4,3, names = ('batch_one', 'batch_two', 'batch_three', 'batch_four', 'batch_five', 'batch_six', 'A', 'B'))
#y = torch.randn(        1,5,3,2, names = ('batch_five', 'batch_six', 'C', 'D'))

#assert checkBroadcastable(x, y)

#x = torch.empty(5, 2, 4, 1)
#y = torch.empty(   3, 1, 1)

#assert not checkBroadcastable(x, y)



# -------------------------------------------

# Fold left
# Source: https://hyp.is/b9hFGg8lEeutuBP4rylAmQ/www.burgaud.com/foldl-foldr-python

foldLeft = lambda f, acc, xs: reduce(f, xs, acc)

# Tests
#assert foldLeft(operator.sub, 0, [1,2,3]) == -6
#assert foldLeft(operator.add, 'L', ['1', '2', '3']) == 'L123'


# Lambda way: 
# foldr = lambda f, acc, xs: reduce(lambda x, y: f(y, x), xs[::-1], acc)
# Function (readable) way: 
def flip(f):
    #TODO is this correct or same as below? 
    # return lambda x, y: f(y, x)
    @functools.wraps(f)
    def newFunc(x, y):
        return f(y, x)
    return newFunc 


# TODO: understand fold right better
def foldRight(f, acc, xs):
    return reduce(flip(f), reversed(xs), acc)

# Tests
#assert foldRight(operator.sub, 0, [1,2,3]) == 2
#assert foldRight(operator.add, 'R', ['1', '2', '3']) == '123R'




# Matrix Util area (symbolic) ---------------------------------



'''
GOAL: removes the UnevaluatedExpr from output of freeze() . As a result, we can have the result of polarize() (frozen output) equal the original expression, without having the equal() method blocked because of different Add vs. MatAdd and UnevaluateExpr constructors in the way.

EXAMPLE: 

equal(polarize(.., expr), expr)  -------------------------> FALSE
equal( removeUnevaluations(polarize(.., expr)), expr) ----> TRUE



TODO: seems to keep the expression unevaluated but not sure yet, and not sure if it is ALWAYS true that all the Add -> MatAdd, Mul -> MatMul, Pow -> MatPow (if not then may need ot make separate function to replace those constructors too)
'''
def removeUnevaluations(expr: MatrixExpr) -> MatrixExpr: 
    oldExprStr: str = srepr(expr)

    newExprStr: str = oldExprStr.replace("UnevaluatedExpr", "")

    # Create the symbols dict
    symbolsDict: Dict[str, MatrixSymbol] = dict(list(map(lambda s: (str(s), s), expr.free_symbols)) )

    return parse_expr(newExprStr, local_dict= symbolsDict)





'''
GOAL: remove MatPow( group, Integer(1)) to just group expression, but no need to do that if the 'group' is just a matrixsymbol or a symbol of any kind

REASON: equal(res, givenExpr) sometimes doesn't work and it is because of the matpow of 1 around an expression group, but subtraction goes to zero easily when matpow of 1 is around a symbol, so leaving that alone 
'''
def removeMatPowOfOne(expr: MatrixExpr) -> MatrixExpr:

    # Filter out nums or syms
    if isNum(expr) or isSym(expr):
        return expr 
        
    if isinstance(expr, MatPow):
        
        (base, expo) = expr.args 

        if expo == Number(1):
            return base # since X^1 == X
        
        # else recurse and wrap with the matpow power
        return MatPow( base = removeMatPowOfOne(base) , exp = expo)
    
    # else is a general non-matpow constructor, can just call 'args' and squash the arg list using * arg operator
    Constr = getConstr(expr)
    # Must apply this function to each argument
    argsOneOut: List[MatrixExpr] = list(map(lambda theArg: removeMatPowOfOne(theArg), expr.args))

    return Constr(*argsOneOut)








# SOURCE OF CODE = https://stackoverflow.com/a/52196005
def expandMatmul(expr: MatrixExpr) -> MatrixExpr:
    #import itertools
#    from sympy import preorder_traversal
    for a in preorder_traversal(expr):
        if isinstance(a, MatMul) and any(isinstance(f, MatAdd) for f in a.args):
            terms = [f.args if isinstance(f, MatAdd) else [f] for f in a.args]
            expanded = MatAdd(*[MatMul(*t) for t in itertools.product(*terms)])
            if a != expanded:
                expr = expr.xreplace({a: expanded})
                return expandMatmul(expr)
    return expr


def equal(mat1: MatrixExpr, mat2: MatrixExpr) -> bool:

    if isinstance(mat1, MatrixExpr) and isinstance(mat2, MatrixExpr):
        if mat1.shape == mat2.shape:
            zeroMat = ZeroMatrix(*mat1.shape)

            # Sometimes the subtraction results in one side of the expression having MatPow(expr, Integer(1)) which impedes simplification. Must remove that and see if results to zero matrix. If not, send to expandMatmul
            subtracted = removeMatPowOfOne(mat1 - mat2).doit()

            return subtracted == zeroMat or expandMatmul(subtracted).doit() == zeroMat

    # else check if they are traces
    elif mat1.is_Trace and mat2.is_Trace:
        return equal(mat1.arg, mat2.arg)

    return False 