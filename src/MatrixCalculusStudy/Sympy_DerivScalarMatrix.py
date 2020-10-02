# %% codecell
from sympy import diff, sin, exp, symbols, Function, Matrix, MatrixSymbol, FunctionMatrix, derive_by_array





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

# %% markdown [markdown]
# ### Derivative of Scalar Function of a Matrix with Respect to the Matrix
# Let $X = \{ x_{ij} \}$ be a matrix of order $m \times n$ and let
# $$
# y = f(X)
# $$
# be a scalar function of $X$, so $y \in \mathbb{R}$ and $f: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}$,
# Then we can define the **derivative of y with respect to $X$** as the following matrix of order $m \times n$:
# $$
# \Large
# \begin{aligned}
# \frac{\partial y}{\partial X} = \begin{pmatrix}
#    \frac{\partial y}{\partial x_{11}} & \frac{\partial y}{\partial x_{12}} & ... & \frac{\partial y}{\partial x_{1n}} \\
#    \frac{\partial y}{\partial x_{21}} & \frac{\partial y}{\partial x_{22}} & ... & \frac{\partial y}{\partial x_{2n}} \\
#    \vdots & \vdots & & \vdots \\
#    \frac{\partial y}{\partial x_{m1}} & \frac{\partial y}{\partial x_{m2}} & ... & \frac{\partial y}{\partial x_{mn}} \\
# \end{pmatrix}
# = \Bigg\{ \frac{\partial y}{\partial x_{ij}} \Bigg\}
# \end{aligned}
# $$
# The matrix $\frac{\partial y}{\partial X}$ is called the **gradient matrix**.
# %% codecell
derive_by_array(Y[0,0], X)



# %% markdown [markdown]
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
# \begin{aligned}
# \frac{\partial Y}{\partial x} = \begin{pmatrix}
#    \frac{\partial Y}{\partial x} & \frac{\partial Y}{\partial x} & ... & \frac{\partial Y}{\partial x} \\
#    \frac{\partial Y}{\partial x} & \frac{\partial Y}{\partial x} & ... & \frac{\partial Y}{\partial x} \\
#    \vdots & \vdots & & \vdots \\
#    \frac{\partial Y}{\partial x} & \frac{\partial Y}{\partial x} & ... & \frac{\partial Y}{\partial x} \\
# \end{pmatrix}
# = \Bigg\{ \frac{\partial y_{ij}}{\partial x} \Bigg\}
# \end{aligned}
# $$
# %% codecell
derive_by_array(Y, X[1-1,2-1])
