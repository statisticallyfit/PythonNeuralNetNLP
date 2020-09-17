# %% codecell
from sympy import Matrix, Symbol, derive_by_array, Lambda, Function, MatrixSymbol
from sympy import myvar
from sympy.abc import x, i, j, a, b

# %% markdown
# Defining myvariable-element matrices $X \in \mathbb{R}^{n \times m}$ and $W \in \mathbb{R}^{m \times p}$:
# %% codecell
def myvar(letter: str, i: int, j: int) -> Symbol:
    letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
    return letter_ij


n,m,p = 3,3,2

X = Matrix(n, m, lambda i,j : myvar('x', i, j)); X
# %% codecell
W = Matrix(m, p, lambda i,j : myvar('w', i, j)); W
# %% codecell
A = MatrixSymbol('X',3,3); Matrix(A)
B = MatrixSymbol('W',3,2)

# %% codecell
v = Lambda((a,b), a*b); v
n = Function('v') #, Lambda((a,b), a*b))
type(v)

# %% codecell
n(X,W)
# %% codecell
n(A,B)
# %% codecell
n(X,W)
# %% codecell
n(X,W).subs({n: v})
# %% codecell
Matrix(n(A,B).subs({n: v}))


# %% codecell
#N = v(X, W); N
N = n(A,B); N
# %% codecell
Nresult = N.subs({n: v, A: X, B:W}); Nresult


# %% codecell
N.diff(N)
# %% codecell
N.diff(X)
# %% codecell
# %% codecell




# %% codecell
# way 2 of declaring S (better way)
sigma = Function('sigma')
sigmaApply = Function("sigma_{apply}") #lambda matrix:  matrix.applyfunc(sigma)
sigmaApply(A)
# %% codecell
sigmaApply_ = lambda matrix: matrix.applyfunc(sigma)
sigmaApply(A).subs({A: X}).replace(sigmaApply, sigmaApply_) # NOTE: subs of functions doesn't work, replace actually evaluates the replaced function!

# %% codecell
N = n(A,B); N
# %% codecell
S = sigmaApply(N); S
# %% codecell
S.subs({n: v})
# %% codecell
S.subs({n: v, A:X, B:W})
# %% codecell
Sresult = S.subs({n: v, A:X, B:W}).replace(sigmaApply, sigmaApply_); Sresult

#S = sigmaApply(n(A,B).subs({n: v, A: X, B:W})) # composing
# %% codecell
# CAN even replace elements after have done an operation on them!!! replacing n_21 * 2 with the number 4.
Sresult.subs({Nresult[0,0]: 3}).replace(sigma, lambda x: 2*x).replace(Nresult[2,1]*2, 4)




# %% codecell
lambd = Function("lambda")
lambd_ = lambda matrix : sum(matrix)

L = lambd(S); L

# %% codecell
L.subs({n: v, A:X, B:W}).replace(sigmaApply, sigmaApply_)
# %% codecell
L.subs({n: v, A:X, B:W}).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)

L_ = lambd_(S_); L_
# %% codecell
L_.subs({n: v})
L_.diff(n)




# %% codecell
from sympy import symbols, Derivative

x, y, r, t = symbols('x y r t') # r (radius), t (angle theta)
f, g, h = symbols('f g h', cls=Function)
h = g(f(x))
Derivative(h, f(x)).doit()


# %% codecell

# %% codecell
n_ij = Function('n_ij')(A,B); n_ij # (N[0,0]); n_ij

n_ij.args
#Nresult[0,0].args
# %% codecell
# sigma(n_ij).diff(n_ij).replace(n_ij, N[0,0]) # ERROR cannot deriv wi.r.t to the expression w11*x11 + ...

sigma(n_ij).diff(n_ij)
# %% codecell
sigma(n_ij).diff(n_ij).subs({n_ij : Nresult[0,0]})

# %% codecell
# ### WAY 2:
n_11 = Function('n_11')(Nresult[0,0]); n_11

# %% codecell
sigma(n_11)
# %% codecell
sigma(n_11).subs({n_11 : n_11.args[0]})
# %% codecell
sigma(n_11).diff(n_11) #.replace(n_ij, n_ij.args[0])
# %% codecell
sigma(n_11).diff(n_11).subs({n_11 : n_11.args[0]}).subs({X[0,0]:77777})
# %% codecell
sigma(n_11).diff(n_11).subs({n_11 : n_11.args[0]}).replace(n_11.args[0], 23) # same as subs in this case
# %% codecell
sigma(n_11).diff(n_11).subs({n_11 : n_11.args[0]}).replace()
#sigma(n_11).diff(n_11).subs({n_11 : n_11.args[0]}).subs({ksi: 23})

# %% codecell
sigma(n_11).diff(X[0,0])
# %% codecell


# %% codecell
# %% codecell
