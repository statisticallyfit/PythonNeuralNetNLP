# %% codecell
from sympy import diff, sin, exp, symbols, Function, Matrix, MatrixSymbol, FunctionMatrix, derive_by_array



x, y, z = symbols('x y z')
f, g, h = list(map(Function, 'fgh'))

xv = x,y,z
#f(xv).subs({x:1, y:2,z:3})
yv = [f(*xv), g(*xv), h(*xv)]; yv
# %% codecell
Matrix(yv)
# %% codecell
Matrix(xv)

# %% codecell
n,m,p = 3,3,2
X = MatrixSymbol('x', n, m)
Matrix(X)
# %% codecell
m = Matrix([[1,0,2],[3,4,5],[3,3,2]]); m
# %% codecell
X.subs({X: m})
# %% codecell
f(X)
# %% codecell
f(Matrix(X))
# %% codecell
f(X).subs({X: m})
# %% codecell
#f(X).diff(Matrix(X)) # Error non commutative elements in matrix

f(Matrix(X)).diff(Matrix(X))

# %% codecell
from sympy import Derivative
M = Matrix(X)
Derivative(f(X), M)
# %% codecell
#diff(f(X), X) # NOT RUN THIS LINE TOO SLOW
# %% codecell
derive_by_array(f(Matrix(X)), Matrix(X))



# %% codecell
n,m,p = 3,3,2
X = Matrix(MatrixSymbol('x', n,m)); X
# %% codecell
W = Matrix(MatrixSymbol('w', m,p)); W
# %% codecell
X*W
# %% codecell
# %% codecell
# %% codecell
# %% codecell
# %% codecell
