# %% codecell
# SOURCE = https://www.kannon.link/free/2019/10/30/symbolic-matrix-differentiation-with-sympy/
from sympy import diff, symbols, MatrixSymbol, Transpose, Trace, Matrix, Function


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
from sympy import srepr, expand, simplify, collect, factor, cancel, apart

#srepr(sq.args[0])
expand(sq.args[0])
# %% codecell
#diff(sq.args[0], B)
#diff(expand(sq.args[0]), B).doit()
from sympy import Symbol
Xm = Matrix(3,3, lambda i,j : Symbol("x_{}{}".format(i+1,j+1), commutative=True))
Wm = Matrix(3,2, lambda i,j : Symbol("w_{}{}".format(i+1,j+1), commutative=True))

X = MatrixSymbol('X',3,3)
W = MatrixSymbol('W', 3,2);

# %% codecell
diff(X*W, X)
# %% codecell
diff(X*W, X).subs({X:Xm})
# %% codecell
diff(X*W, X).subs({X:Xm}).doit()
# %% codecell
diff(X*W, X).subs({X:Xm}).doit().subs({W:Wm})

# %% codecell
# expand(diff(X*W, X).subs({X:Xm}).doit().subs({W:Wm}))# STUCK doesn't work to expand out from here
#diff(X*W, X).replace(X,Xm)# ERROR so I must use subs instead (noncommutatitve scalars in matrix multiplication not supported)
diff(X*W, X).subs({X:Xm, W:Wm}).doit()

# %% codecell
g,f = symbols('g f', cls = Function)
f(X).replace(X, X.T).diff(X).replace(X.T, X)
# %% codecell
g(f(X)).replace(X, X.T).diff(X).replace(X.T, X)
# %% codecell
f(X,W).replace(X,X.T).diff(X)
2+2
# %% codecell
# %% codecell
# %% codecell
# %% codecell
# %% codecell




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
# %% codecell
type(A.T)
# %% codecell
type(Z + A)
# %% codecell
type(A*1)
# %% codecell
type(A)
# %% codecell
type(A*B)
# %% codecell
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
from sympy.abc import i,j,x,a,b,c

h = g(f(A.T))

h
# %% codecell

diff(h, A).replace(A.T,A)
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
Matrix(M)
# %% codecell
Matrix(A)
# %% codecell
g, f = symbols('g f', cls = Function)

#__ = lambda mat: mat.T # transposes matrix symbol

diff( g(f(M,b)), b)

# %% codecell
diff( g(f(M,b)), b).replace(M, A)
# %% codecell
Ms = MatrixSymbol('M',2,2)
Ds = MatrixSymbol('D',2,2)
M = Matrix(2,2, lambda i,j: Symbol("m_{}{}".format(i+1,j+1)))
D = Matrix(2,2, lambda i,j: Symbol("d_{}{}".format(i+1,j+1)))

diff( g(f(M, D)), D )
# %% codecell
diff( g(f(M, D)), D ).replace(D, Ds).replace(M, Ms)
# %% codecell
diff(Ds,Ds).replace(Ds,D).doit()

# %% codecell
#diff( g(f(Ms, Ds.T)), Ds )#.replace(Ds.T, Ds)
# %% codecell
# %% codecell
# %% codecell
