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
from sympy import Symbol

def var(letter: str, i: int, j: int) -> Symbol:
     letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
     return letter_ij

def func(i, j):
     y_ij = Function('y_{}{}'.format(i+1,j+1))(*X)
     return y_ij


n,m,p = 3,3,2

X = Matrix(n, m, lambda i, j: var('x', i, j)); X
# %% codecell
#Y = MatrixSymbol(Function('y'), 2, 3); Matrix(Y)
#M = MatrixSymbol('M',2,2); Matrix(M)
#Y = Matrix(m, p, lambda i,j: Function('y_{}{}'.format(i+1,j+1))(X) ); Y

Y = Matrix(m, p, lambda i,j:  func(i, j)); Y

# %% markdown
# ### Derivative of Scalar Function of a Matrix with Respect to the Matrix
# Let $X = \{ x_{ij} \}$ be a matrix of order $m \times n$ and let 
# $$
# y = f(X)
# $$ 
# be a scalar function of $X$, so $y \in \mathbb{R}$ and $f: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}$, 
# Then we can define the **derivative of y with respect to $X$** as the following matrix of order $m \times n$: 
# $$
# \Large
# \frac{\partial y}{\partial X} = \begin{pmatrix}
#    \frac{\partial y}{\partial x_{11}} & \frac{\partial y}{\partial x_{12}} & ... & \frac{\partial y}{\partial x_{1n}} \\
#    \frac{\partial y}{\partial x_{21}} & \frac{\partial y}{\partial x_{22}} & ... & \frac{\partial y}{\partial x_{2n}} \\
#    \vdots & \vdots & & \vdots \\
#    \frac{\partial y}{\partial x_{m1}} & \frac{\partial y}{\partial x_{m2}} & ... & \frac{\partial y}{\partial x_{mn}} \\
# \end{pmatrix}
# 
# = \Bigg\{ \frac{\partial y}{\partial x_{ij}} \Bigg\}
# $$
# The matrix $\frac{\partial y}{\partial X}$ is called the **gradient matrix**. 
# %% codecell
derive_by_array(Y[0,0], X)



# %% markdown
# ### Derivative of Matrix With Respect to Scalar Element of Matrix
# Let $X = \{ x_{ij} \}$ be a matrix of order $m \times n$ and let 
# $$
# y = f(X)
# $$ 
# be a scalar function of $X$, so $y \in \mathbb{R}$ and $f: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}$, 
# 
# Also let the matrix $Y = \{y_{ij}(X) \}$ be of size $p \times q$. 
# 
# Then we can define the **derivative of $Y$ with respect to an element $x$ in $X$** as the following matrix of order $p \times q$: 
# $$
# \Large
# \frac{\partial Y}{\partial x} = \begin{pmatrix}
#    \frac{\partial Y}{\partial x} & \frac{\partial Y}{\partial x} & ... & \frac{\partial Y}{\partial x} \\
#    \frac{\partial Y}{\partial x} & \frac{\partial Y}{\partial x} & ... & \frac{\partial Y}{\partial x} \\
#    \vdots & \vdots & & \vdots \\
#    \frac{\partial Y}{\partial x} & \frac{\partial Y}{\partial x} & ... & \frac{\partial Y}{\partial x} \\
# \end{pmatrix}
# 
# = \Bigg\{ \frac{\partial y_{ij}}{\partial x} \Bigg\}
# $$
# %% codecell
derive_by_array(Y, X[1-1,2-1])




# %% codecell
# GOT IT this is the definition of gradient matrix (matrix of partial derivatives or dY/dX)
D = derive_by_array(Y, X); D
# %% codecell
D.subs({Y[0,0]: X[0,0]**2 + X[1,0]}).doit()
# %% codecell
Y.diff(X) ## GOT IT



# %% codecell
Yval = Y.subs({Y[0,0]: X[0,0]**2 + X[0,1]*X[1,0] - X[1,1],
        Y[0,1]: X[1,1]**3 + 4* X[0,1] + X[0,0] - X[1,0],
        Y[1,0]: X[1,0] * X[0,0] + 3*X[0,1] * X[1,1],
        Y[1,1]: X[1,1] + X[1,0] + X[0,1] + X[0,0],
        Y[2,0]: 2*X[0,0]**2 * X[0,1] * 3*X[1,0] + 4*X[1,1],
        Y[2,1]: 3*X[0,1] - 5*X[1,1] * X[0,0] - X[1,0]**2})

Yval
# %% codecell
DYval = D.subs({Y[0,0]: X[0,0]**2 + X[0,1]*X[1,0] - X[1,1],
        Y[0,1]: X[1,1]**3 + 4* X[0,1] + X[0,0] - X[1,0],
        Y[1,0]: X[1,0] * X[0,0] + 3*X[0,1] * X[1,1],
        Y[1,1]: X[1,1] + X[1,0] + X[0,1] + X[0,0],
        Y[2,0]: 2*X[0,0]**2 * X[0,1] * 3*X[1,0] + 4*X[1,1],
        Y[2,1]: 3*X[0,1] - 5*X[1,1] * X[0,0] - X[1,0]**2})
DYval
# %% codecell
DYval.doit()




# %% codecell
# ### GOAL: testing the A kronecker B rule for diff of Y = AXB
from sympy import Lambda
l, m, n, q = 3, 5, 4, 2

A = Matrix(l, m, lambda i, j: var('a', i, j))
X = Matrix(m, n, lambda i, j: var('x', i, j))
W = Matrix(n, q, lambda i, j: var('w', i, j))
Y = X*W; Y
# %% codecell
from sympy.matrices import zeros
E_12 = zeros(m, n)
E_12[1-1,2-1] = 1
E_12
# %% codecell
Y = X*W; Y
# %% codecell
E_12*W
# %% codecell
derive_by_array(Y, X[0,1])
# %% codecell
assert Matrix(derive_by_array(Y, X[0,1])) == E_12 * W

assert Matrix(derive_by_array(Y, X[0,1])) == Y.diff(X[0,1])
