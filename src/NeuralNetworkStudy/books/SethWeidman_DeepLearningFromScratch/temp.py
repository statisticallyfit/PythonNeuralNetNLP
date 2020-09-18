# %% codecell

from sympy import diff, symbols, MatrixSymbol, Transpose, Trace, Matrix


def squared_frobenius_norm(expr):
    return Trace(expr * Transpose(expr))

k, m, n = symbols('k m n')

X = MatrixSymbol('X', m, k)
W = MatrixSymbol('W', k, n)
Y = MatrixSymbol('Y', m, n)

# Matrix(X)
A = MatrixSymbol('A', 3, 4)
B = MatrixSymbol('B', 4, 2)
C = MatrixSymbol('C', 3, 2)
Matrix(A)

# %% codecell
diff(squared_frobenius_norm(X*W - Y), W)
# %% codecell
sq = squared_frobenius_norm(A*B - C); sq
# %% codecell
diff(squared_frobenius_norm(A*B - C), B)

# %% codecell
sq.args[0]
# %% codecell
type(sq.args[0])
# %% codecell
from sympy import symbols, Function

#h,g,f = symbols('h g f', cls=Function)
f = Function('f')
g = Function('g')
h = g(f(sq.args[0]))
h
# %% codecell
diff(h, B)
# %% codecell
from sympy import Derivative

#h.replace(f, Trace)

# %% codecell
diff(sq.args[0], B)
# %% codecell
from sympy import Trace


h = f(Trace(sq.args[0]))

diff(h, B)
# %% codecell
h = g(f(A*B))
h

# %% codecell
diff(h, A)
# %% codecell
from sympy import ZeroMatrix
Z = ZeroMatrix(3,4); Z
Matrix(Z)
type(A.T)
type(Z + A)
type(A*1)
type(A)
type(A*B)
from sympy.matrices.expressions.matexpr import MatrixExpr

#Matrix(MatrixExpr(A)) # ERROR
# %% codecell
# %% codecell
# diff(h, A) # WHAT THIS IS STILL BAD

# This is why:
assert type(A.T) != type(A.T.T)
#h = g(f(Z + A))
#D = MatrixSymbol('D', 3,4)

#ad = A+D
h = g(f(__(A)))

h
# %% codecell

diff(h, A).replace(__(A),A)
# %% codecell
diff(A.T, A).replace(A.T, A)

# %% codecell
diff(A.T, A).replace(A, Matrix(A))#.doit()
# %% codecell
diff(A.T, A).replace(A, Matrix(A)).doit()


# %% codecell
from sympy import Symbol 
from sympy.abc import b

#A = MatrixSymbol('A', 3,4)
M = Matrix(3,4, lambda i,j : Symbol('x_{}{}'.format(i+1,j+1)))
g, f = symbols('g f', cls = Function)

__ = lambda mat: mat.T # transposes matrix symbol

h = g(f(A,b))
h
# %% codecell


# %% codecell
# %% codecell
# %% codecell
# %% codecell
