# %% codecell
from sympy import Matrix, Symbol, derive_by_array, Lambda, Function, MatrixSymbol
from sympy.abc import x, i, j, a, b

# %% markdown
# Defining variable-element matrices $X \in \mathbb{R}^{n \times m}$ and $W \in \mathbb{R}^{m \times p}$:
# %% codecell
def var(letter: str, i: int, j: int) -> Symbol:
    letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
    return letter_ij


n,m,p = 3,3,2

X = Matrix(n, m, lambda i,j : var('x', i, j)); X
# %% codecell
W = Matrix(m, p, lambda i,j : var('w', i, j)); W

# %% codecell
v = Lambda((a,b), a*b); v

# %% codecell
N = v(X, W); N



# %% codecell
# way 2 of declaring S (better way)
sigma = Function('sigma')
sigmaApply = lambda matrix:  matrix.applyfunc(sigma)

S = sigmaApply(v(X,W)) # composing
S

# %% codecell
lambd = lambda matrix : sum(matrix)

L = lambd(sigmaApply(v(X, W)))
L


# %% codecell
from sympy import symbols, Derivative

x, y, r, t = symbols('x y r t') # r (radius), t (angle theta)
f, g, h = symbols('f g h', cls=Function)
h = g(f(x))
Derivative(h, f(x)).doit()


# %% codecell
A = MatrixSymbol('X',3,3); Matrix(A)
B = MatrixSymbol('W',3,2)
# %% codecell
n_ij = Function('n_ij')(A,B); n_ij # (N[0,0]); n_ij
n_ij.args

N[0,0].args
# %% codecell
# sigma(n_ij).diff(n_ij).replace(n_ij, N[0,0]) # ERROR cannot deriv wi.r.t to the expression w11*x11 + ...

# %% codecell
sigma(n_ij).diff(A)
# %% codecell
sigma(n_ij).diff(n_ij)
# %% codecell
sigma(n_ij).diff(Matrix(X))

# %% codecell

sigma(n_ij).diff(n_ij)
# %% codecell
sigma(n_ij).diff(n_ij).subs({n_ij: N[0,0]})






# %% codecell
# ### WAY 2:
n_ij = Function('n_11')(N[0,0]); n_ij

# %% codecell
sigma(n_ij)
# %% codecell
sigma(n_ij).subs({n_ij : n_ij.args[0]})
# %% codecell
sigma(n_ij).diff(n_ij) #.replace(n_ij, n_ij.args[0])
# %% codecell
sigma(n_ij).diff(n_ij).subs({n_ij : n_ij.args[0]})
# %% codecell
sigma(n_ij).diff(X[0,0])
# %% codecell
sigma(n_ij).diff(X[0,0]).subs({n_ij: n_ij.args[0]})

# %% codecell
# %% codecell
