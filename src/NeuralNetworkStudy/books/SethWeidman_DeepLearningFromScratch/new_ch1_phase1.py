# %% codecell
from sympy import Matrix, Symbol, derive_by_array, Lambda, symbols, Derivative, diff
from sympy.abc import x, y, i, j, a, b

# %% markdown [markdown]
#
# Defining variable-element matrices $X \in \mathbb{R}^{n \times m}$ and $W \in \mathbb{R}^{m \times p}$:
# %% codecell
def var(letter: str, i: int, j: int) -> Symbol:
    letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
    return letter_ij


n,m,p = 3,3,2

X = Matrix(n, m, lambda i,j : var('x', i, j)); X
# %% codecell
W = Matrix(m, p, lambda i,j : var('w', i, j)); W

# %% markdown [markdown]
# Defining $N = \nu(X, W) = X \times W$
#
# * $\nu : \mathbb{R}^{(n \times m) \times (m \times p)} \rightarrow \mathbb{R}^{n \times p}$
# * $N \in \mathbb{R}^{n \times p}$
# %% codecell
v = Lambda((a,b), a*b); v

# %% codecell
N = v(X, W); N

# %% markdown [markdown]
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

# %% markdown [markdown]
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


# %% markdown [markdown]
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
n11
# %% codecell
s_ij = Function('s_ij')
sig = Function('sig')(x)


# %% codecell

# KEY: got not expecting UndefinedFunction error again here too
#S_ij = Matrix(3, 2, lambda i,j: Function('s_{}{}'.format(i+1,j+1))(Function('{}'.format(N[i,j]))))



# %% codecell
#S_ij[0,0](sympify(N[0,0])).diff(sympify(N[0,0]))
F = 3*x*y

xy = Symbol('{}'.format(F))
xy.subs({x:3})
sympify(xy).subs({x:3})

# %% markdown [markdown]
# Sympy Example of trying to differentiate with respect to an **expression** not just a variable.
# %% codecell
from sympy.abc import t

F = Function('F')
f = Function('f')
U = f(t)
V = U.diff(t)

direct = F(t, U, V).diff(U); direct

# %% codecell
F(t,U,V)
# %% codecell
F(t,U,V).subs(U,x)
# %% codecell
F(t,U,V).subs(U,x).diff(x)
# %% codecell
F(t,U,V).subs(U,x).diff(x).subs(x, U)
# %% codecell
indirect = F(t,U,V).subs(U, x).diff(x).subs(x,U); indirect
# %% codecell
F = Lambda((x,y), 3*x* y)
F(1,2)
# %% codecell
U = x*y
G = 3*x*y
xy
# %% codecell
F.diff(xy)
# %% codecell
# derive_by_array(S, N) # ERROR

# %% codecell
s11 = S[0,0]
s11
# %% codecell

#s11.diff(n11)

# %% codecell
derive_by_array(L, S)


# %% codecell

x, y, r, t = symbols('x y r t') # r (radius), t (angle theta)
f, g, h = symbols('f g h', cls=Function)
h = g(f(x))
Derivative(h, f(x)).doit()




# %% codecell
h.args[0]
h.diff(h.args[0])


# %% codecell
S = sigmaApply(v(X,W)); S

# %% codecell
from sympy.abc import n

n11 = (X*W)[0,0]
m = lambda mat1, mat2: sympify(Symbol('{}'.format((mat1 * mat2)[0,0] )))
s = sigma(m(X,W)); s
# %% codecell
s.subs({W[0,0]: 14}) # doesn't work to substitute into an undefined function
# %% codecell
Derivative(s, m(X,W)).doit()
# %% codecell

#s11 = Function('s_{11}')(n11); s11
#sigma(n11).diff(n11)

#s11.diff(n11)
sigma(n11)
# %% codecell
# ERROR HERE TOO
type(sigma(n11).args[0])
# %% codecell
type(n11)
# %% codecell
#sigma(n11).diff(sigma(n11).args[0]) ## ERROR
# %% codecell

# %% codecell
b = Symbol('{}'.format(n11))
ns_11 = Function(b, real=True)
ns_11


# ERROR cannot diff wi.r. to undefinedfunction
# sigma(n11).diff(ns_11)


#
#sigma(b).diff(b).subs({b:1})


# %% codecell
f, g = symbols('f g', cls=Function)
xy = Symbol('x*y'); xy
#sympify(xy).subs({x:2, y:4})
f(g(x,y)).diff(xy)
# %% codecell
# TODO SEEM to have got the expression but it is not working since can't substitute anything .... ???
f(xy).diff(xy).subs({x:2})
# %% codecell
Function("x*y")(x,y)
xyf = lambdify([x,y],xy)
xyf(3,4)
f(g(xy)).diff(xy)
#

# %% codecell
xyd = Derivative(x*y, x*y,0).doit();xyd

#Derivative(3*xyd, xyd, 1).doit() ### ERROR can't calc deriv w.r.t to x*y

# %% codecell
#derive_by_array(S, N)





# %% codecell
# %% codecell
