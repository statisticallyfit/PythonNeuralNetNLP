# %% codecell
from sympy import Matrix, Symbol, derive_by_array, Lambda
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

# %% markdown
# Defining $N = \nu(X, W) = X \times W$
#
# * $\nu : \mathbb{R}^{(n \times m) \times (m \times p)} \rightarrow \mathbb{R}^{n \times p}$
# * $N \in \mathbb{R}^{n \times p}$
# %% codecell
v = Lambda((a,b), a*b); v

# %% codecell
N = v(X, W); N

# %% markdown
#
# Defining $S = \sigma_{\text{apply}}(N) = \sigma_{\text{apply}}(\nu(X,W)) = \sigma_\text{apply}(X \times W) = \Big \{ \sigma(XW_{ij}) \Big\}$.
#

# Assume that $\sigma_{\text{apply}} : \mathbb{R}^{n \times p} \rightarrow \mathbb{R}^{n \times p}$ while $\sigma : \mathbb{R} \rightarrow \mathbb{R}$, so the function $\sigma_{\text{apply}}$ takes in a matrix and returns a matrix while the simple $\sigma$ acts on the individual elements $N_{ij} = XW_{ij}$ in the matrix argument $N$ of $\sigma_{\text{apply}}$.
#
# * $\sigma : \mathbb{R} \rightarrow \mathbb{R}$
# * $\sigma_\text{apply} : \mathbb{R}^{n \times p} \rightarrow \mathbb{R}^{n \times p}$
# * $S \in \mathbb{R}^{n \times p}$
# %% codecell
from sympy import Function

# Nvec = Symbol('N', commutative=False)

sigma = Function('sigma')
sigma(N[0,0])

# %% codecell
# way 1 of declaring S
S = N.applyfunc(sigma); S
#type(S)
#Matrix(3, 2, lambda i, j: sigma(N[i,j]))
# %% codecell
# way 2 of declaring S (better way)
sigmaApply = lambda matrix:  matrix.applyfunc(sigma)

sigmaApply(N)

# %% codecell
sigmaApply(X**2) # can apply this function to any matrix argument.
# %% codecell
S = sigmaApply(v(X,W)) # composing
S

# %% markdown
# Defining $L = \Lambda(S) = \Lambda(\sigma_\text{apply}(\nu(X,W))) = \Lambda \Big(\Big \{ \sigma(XW_{ij}) \Big\} \Big)$. In general, let the function be defined as:
# $$
# \begin{align}
# L &= \Lambda \begin{pmatrix}
#    \sigma(XW_{11}) & \sigma(XW_{12}) & ... & \sigma(XW_{1p}) \\
#    \sigma(XW_{21}) & \sigma(XW_{22}) & ... & \sigma(XW_{2p}) \\
#    \vdots & \vdots & & \vdots \\
#    \sigma(XW_{n1}) & \sigma(XW_{n2}) & ... & \sigma(XW_{np})
# \end{pmatrix} \\
#
# &= \sum_{i=1}^p \sum_{j = 1}^n  \sigma(XW_{ij}) \\
#
# &= \sigma(XW_{11}) + \sigma{XW_{12}} + ... + \sigma(XW_{np})
# \end{align}
# $$
# * $\Lambda: \mathbb{R}^{n \times p} \rightarrow \mathbb{R}$
# * $L \in \mathbb{R}$
# %% codecell
lambdaF = lambda matrix : sum(matrix)
lambdaF(S)
# %% codecell
L = lambdaF(sigmaApply(v(X, W)))
L
#L = lambda mat1, mat2: lambdaF(sigmaApply(v(mat1, mat2)))
#L(X, W)


# %% markdown
# %% codecell
#derive_by_array(L, X)
# %% codecell
derive_by_array(L, S)
# %% codecell
from sympy import sympify, lambdify
n = lambdify((X[0,0],X[0,1],X[0,2],W[0,0],W[1,0],W[2,0]), N[0,0])
n(1,2,3,4,3,2)

# %% codecell
f = Function('f') #(sympify(N[0,0]))
f(N[0,0])
# %% codecell
f(N[0,0]).diff(X[0,0])




# %% codecell
n = v(X,W); n
n11 = Function('{}'.format(n[0,0]))
S[0,0]
s11 = S[0,0]
s11.diff(n11)
# %% codecell
makeFunc = lambda matElem: Function('{}'.format(matElem))
makeFuncOfMatElems = lambda mat: mat.applyfunc(makeFunc)
NN = makeFuncOfMatElems(N)
NN
# %% codecell
SS = sigmaApply(NN); SS
# %% codecell
S[0,0].arg
L = lambdaF(SS)
# %% codecell
derive_by_array(L, S)
# %% codecell
from sympy import symbols
x, y, r, t = symbols('x y r t') # r (radius), t (angle theta)
f, g, h = symbols('f g h', cls=Function)
h = g(f(x))
Derivative(h, x).doit()
# %% codecell
derive_by_array(S, N)



# %% codecell
# %% codecell
