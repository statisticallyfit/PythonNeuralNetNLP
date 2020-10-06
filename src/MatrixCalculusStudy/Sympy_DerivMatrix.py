
# %% [markdown]
# # Review: Derivative of a Matrix with Respect to a Matrix


# %% codecell
from IPython.display import display

### TODO: how to make sympy latex render automatically in VSCode editor, without having to call IPython's display?
from sympy.interactive import init_printing
init_printing(pretty_print=True, use_latex='mathjax')
# init_printing(use_unicode=False, pretty_print=True, latex_mode="equation", use_latex="mathjax")
#init_printing(pretty_print=True)
#init_session(pretty_print=True)
#init_printing(use_latex="mathjax", latex_mode="equation", pretty_print=True)



from sympy import diff, sin, exp, symbols, Function, Matrix, MatrixSymbol, FunctionMatrix, derive_by_array, Derivative


from sympy import Symbol
from sympy import cos

def var(letter: str, i: int, j: int) -> Symbol:
     letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
     return letter_ij

def func(i, j):
     y_ij = Function('y_{}{}'.format(i+1,j+1))(*X)
     return y_ij


n,m,p = 3,3,2

X = Matrix(n, m, lambda i, j: var('x', i, j))

## TODO: when in vscode editor will need to use IPython's display method to display the latex sympy output or else it doesn't work (seems to be a recent issue).
display(X)
# X


# %% 
#Y = MatrixSymbol(Function('y'), 2, 3); Matrix(Y)
#M = MatrixSymbol('M',2,2); Matrix(M)
#Y = Matrix(m, p, lambda i,j: Function('y_{}{}'.format(i+1,j+1))(X) ); Y

Y = Matrix(m, p, lambda i,j:  func(i, j))
#Y
display(Y)



# %% [markdown]
# ### Derivative of Matrix With Respect a Matrix
# Let $X = \{ x_{ij} \}$ be a matrix of order $m \times n$ and let
# $$
# y = f(X)
# $$
# be a scalar function of $X$, so $y \in \mathbb{R}$ and $f: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}$,
#
# Also let the matrix $Y = \{y_{ij}(X) \}$ be of size $p \times q$.
#
# Then we can define the **derivative of $Y$ with respect to $X$** as the following matrix of order $mp \times nq$:
#
# $$
# \Large
# \frac{\partial Y}{\partial X}
# = \begin{pmatrix}
#    \frac{\partial Y}{\partial x_{11}} & \frac{\partial Y}{\partial x_{12}} & ... & \frac{\partial Y}{\partial x_{1n}} \\
#    \frac{\partial Y}{\partial x_{21}} & \frac{\partial Y}{\partial x_{22}} & ... & \frac{\partial Y}{\partial x_{23}} \\
#    \vdots & \vdots & & \vdots \\
#    \frac{\partial Y}{\partial x_{m1}} & \frac{\partial Y}{\partial x_{m2}} & ... & \frac{\partial Y}{\partial x_{mn}} \\
# \end{pmatrix}
# = \Bigg\{ \frac{\partial y_{ij}}{\partial x_{lk}} \Bigg\}
# $$

# %% codecell
Yelem = Matrix(m, p, lambda i, j: var('y', i, j))
display(Yelem)
# %% codecell
import itertools

elemToFuncArgsD = dict(itertools.chain(*[[(Yelem[i, j], Y[i,j]) for j in range(p)] for i in range(m)]))

elemToFuncArgs = list(elemToFuncArgsD.items())

funcArgsToElemD = {v : k for k, v in elemToFuncArgsD.items()}

funcArgsToElem = list(funcArgsToElemD.items())

# Matrix(funcArgsToElem)
display(Matrix(funcArgsToElem))
# %% 
# GOT IT this is the definition of gradient matrix (matrix of partial derivatives or dY/dX)
D = derive_by_array(Y, X)

# %% codecell
display(D)

# %% codecell
#D.subs(funcArgsToElemD)

# NOTE using substituion here makes output shorter (don't need to see all those x_ij arguments, just the function name y_ij)
display(D.subs(funcArgsToElemD))
# %% codecell

# NOTE: interesting: the partial derivative symbol changes to simple 'd' when substituting the y_ij without arguments ... so sympy recognizes it is not differentiating a multivariable functino anymore. 
#D.replace(Y[0,0], Yelem[0,0])
display(D.replace(Y[0,0], Yelem[0,0]))
# %% codecell
#D[0,0][0,0].subs(Y[0,0], Yelem[0,0])
display(D[0,0][0,0].subs(Y[0,0], Yelem[0,0]))
# %% codecell
#Derivative(Yelem[0,0], X[0,0])
display(Derivative(Yelem[0,0], X[0,0]))
# %% codecell
#Derivative(Y, X).doit()
display(Derivative(Y, X).doit())
# %% codecell
#D.subs({Y[0,0]: X[0,0]**2 + X[1,0]}).doit()
display(D.subs({Y[0,0]: X[0,0]**2 + X[1,0]}).doit())
# %% codecell
#Y.diff(X) ## GOT IT
display(Y.diff(X))


# %% codecell
Yval = Y.subs({Y[0,0]: X[0,0]**2 + X[0,1]*X[1,0] - X[1,1],
        Y[0,1]: X[1,1]**3 + 4* X[0,1] + X[0,0] - X[1,0],
        Y[1,0]: X[1,0] * X[0,0] + 3*X[0,1] * X[1,1],
        Y[1,1]: X[1,1] + X[1,0] + X[0,1] + X[0,0],
        Y[2,0]: 2*X[0,0]**2 * X[0,1] * 3*X[1,0] + 4*X[1,1],
        Y[2,1]: 3*X[0,1] - 5*X[1,1] * X[0,0] - X[1,0]**2})

#Yval
display(Yval)
# %% codecell
DYval = D.subs({Y[0,0]: X[0,0]**2 + X[0,1]*X[1,0] - X[1,1],
        Y[0,1]: X[1,1]**3 + 4* X[0,1] + X[0,0] - X[1,0],
        Y[1,0]: X[1,0] * X[0,0] + 3*X[0,1] * X[1,1],
        Y[1,1]: X[1,1] + X[1,0] + X[0,1] + X[0,0],
        Y[2,0]: 2*X[0,0]**2 * X[0,1] * 3*X[1,0] + 4*X[1,1],
        Y[2,1]: 3*X[0,1] - 5*X[1,1] * X[0,0] - X[1,0]**2})
display(DYval)
# %% codecell
display(DYval.doit())




# %% codecell
# ### GOAL: testing the A kronecker B rule for diff of Y = AXB
from sympy import Lambda
l, m, n, q = 3, 5, 4, 2

A = Matrix(l, m, lambda i, j: var('a', i, j))
X = Matrix(m, n, lambda i, j: var('x', i, j))
W = Matrix(n, q, lambda i, j: var('w', i, j))
Y = X*W

#Y
display(Y)
# %% codecell
from sympy.matrices import zeros
E_12 = zeros(m, n)
E_12[1-1,2-1] = 1
#E_12
display(E_12)

# %% codecell
#E_12*W
display(E_12*W)
# %% codecell
#derive_by_array(Y, X[0,1])
display(derive_by_array(Y, X[0,1]))
# %% codecell
assert Matrix(derive_by_array(Y, X[0,1])) == E_12 * W

assert Matrix(derive_by_array(Y, X[0,1])) == Y.diff(X[0,1])

# %%
