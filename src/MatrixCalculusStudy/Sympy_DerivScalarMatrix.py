# %% markdown
# # Review: Derivative of a Scalar Function with Respect to a Matrix

# %% codecell
from sympy import diff, sin, exp, symbols, Function, Matrix, MatrixSymbol, FunctionMatrix, derive_by_array, Symbol

# TODO Need this now when in VSCode otherwise sympy math latex won't render. Need to upgrade vscode perhaps they fix it? (see evernote for link)
from IPython.display import display



# %% codecell
from sympy import Symbol

def var(letter: str, i: int, j: int) -> Symbol:
     letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
     return letter_ij

def func(i, j):
     y_ij = Function('y_{}{}'.format(i+1,j+1))(*X)
     return y_ij


n,m,p = 3,3,2

X = Matrix(n, m, lambda i, j: var('x', i, j))
#X
display(X)
# %% codecell

Yfunc = Matrix(m, p, lambda i,j:  func(i, j))
#Y
display(Yfunc)


# %% codecell
Yelem = Matrix(m, p, lambda i, j: var('y', i, j))
display(Yelem)
# %% codecell
import itertools

elemToFuncArgsD = dict(itertools.chain(*[[(Yelem[i, j], Yfunc[i,j]) for j in range(p)] for i in range(m)]))

elemToFuncArgs = list(elemToFuncArgsD.items())

funcArgsToElemD = {v : k for k, v in elemToFuncArgsD.items()}

funcArgsToElem = list(funcArgsToElemD.items())

# Matrix(funcArgsToElem)
display(Matrix(funcArgsToElem))



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
#derive_by_array(Y
#[0,0], X)
#display(derive_by_array(Y[0,0], X))
derivScalarByMatrix = derive_by_array(Yfunc[0,0], X)

display(derivScalarByMatrix.subs(funcArgsToElemD))


# %% markdown
# ### Derivative of Matrix With Respect to Scalar Element of Matrix
# Let $X = \{ x_{ij} \}$ be a matrix of order $m \times n$ and let
# $$
# y = f(X)
# $$
# be a scalar function of $X$, so $y \in \mathbb{R}$ and $f: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}$,
#
# Also let the matrix $Y
# = \{y_{ij}(X) \}$ be of size $p \times q$.
#
# Then we can define the **derivative of $Y
#$ with respect to an element $x$ in $X$** as the following matrix of order $p \times q$:
# $$
# \Large
# \begin{aligned}
# \frac{\partial Y
#}{\partial x} = \begin{pmatrix}
#    \frac{\partial Y
#}{\partial x} & \frac{\partial Y
#}{\partial x} & ... & \frac{\partial Y
#}{\partial x} \\
#    \frac{\partial Y
#}{\partial x} & \frac{\partial Y
#}{\partial x} & ... & \frac{\partial Y
#}{\partial x} \\
#    \vdots & \vdots & & \vdots \\
#    \frac{\partial Y
#}{\partial x} & \frac{\partial Y
#}{\partial x} & ... & \frac{\partial Y
#}{\partial x} \\
# \end{pmatrix}
# = \Bigg\{ \frac{\partial y_{ij}}{\partial x} \Bigg\}
# \end{aligned}
# $$


# %% codecell
#derive_by_array(Y#, X[1-1,2-1])
derivMatrixByScalar = derive_by_array(Yfunc, X[1-1,2-1])

#display(derivMatrixByScalar.subs(funcArgsToElemD))
display(derivMatrixByScalar.subs(funcArgsToElemD))



