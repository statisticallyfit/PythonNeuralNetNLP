# %% codecell
from sympy import Matrix, Symbol, derive_by_array, Lambda, Function, MatrixSymbol
from sympy import var
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
# %% codecell
# %% codecell
v = lambda a,b: a*b

vL = Lambda((a,b), a*b)

n = Function('v') #, Lambda((a,b), a*b))

vv = lambda mat1, mat2: Matrix(mat1.shape[0], mat2.shape[1], lambda i, j: Symbol("n_{}{}".format(i+1, j+1))); vv

Nelem = vv(X, W); Nelem

# %% codecell
n(X,W)
# %% codecell
n(A,B)
# %% codecell
n(X,W).replace(n, v) # replace works when v = python lambda
# %% codecell
n(X,W).subs({n: vL}) # subs works when v = sympy lambda
# %% codecell
n(X,W).replace(n, vL)
# %% codecell
n(X,W).subs({n: v})# subs() doesn't work when v is python lambda

# %% codecell
Matrix(n(A,B).subs({n: vL}))


# %% codecell
#N = v(X, W); N
N = n(A,B); N
# %% codecell
N.replace(n, v)
# %% codecell
N.replace(n, v).subs({A: X, B:W}) # replacing ariable values after doing function doesn't make the function apply directly on the values (matrices), need to replace values before the function is replaced, so that the function can act on them while they are given/alive.

# %% codecell
N.subs({n: vL, A:X, B:W})
# %% codecell
Nspec = N.subs({A:X, B:W}).replace(n, v); Nspec
# %% codecell
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

sigmaApply_ = lambda matrix: matrix.applyfunc(sigma)

sigmaApply(A)
# %% codecell
sigmaApply_(A)
# %% codecell
sigmaApply_(X)

# %% codecell
sigmaApply(A).subs({A: X}).replace(sigmaApply, sigmaApply_) # NOTE: subs of functions doesn't work, replace actually evaluates the replaced function!


# %% codecell
S = sigmaApply(N); S
# %% codecell
S.replace(A,X).replace(B,W)

# %% codecell
S.replace(n, v)
# %% codecell
S.subs({A:X, B:W}).replace(n, v)
# %% codecell
Sspec = S.subs({A:X, B:W}).replace(n, v).replace(sigmaApply, sigmaApply_); Sspec
# %% codecell
S.replace(n, vv) #.replace(sigmaApply, sigmaApply_)
# %% codecell
Selem = S.replace(n, vv).replace(sigmaApply, sigmaApply_); Selem
# %% codecell
Selem.subs({Nelem[0,0]: Nspec[0,0], Nelem[2,1]:Nspec[2,1]})
# %% codecell
Selem[0,1].diff(Nelem[0,1])
# %% codecell
Selem[0,1].diff(Nelem[0,1]).subs({Nelem[0,1]: Nspec[0,1]})
# %% codecell
Selem[0,1].diff(Nelem[0,1]).subs({Nelem[0,1]: Nspec[0,1]}).subs({Nspec[0,1] : 23})
# %% codecell
Selem[0,1].diff(Nelem[0,1]).subs({Nelem[0,1] : Nspec[0,1]}).replace(sigma, lambda x: 8*x**3)
# %% codecell
# ### GOT IT: can replace now with expression and do derivative with respect to that expression. 
Selem[0,1].diff(Nelem[0,1]).subs({Nelem[0,1] : Nspec[0,1]}).replace(sigma, lambda x: 8*x**3).doit()

# %% codecell
# CAN even replace elements after have done an operation on them!!! replacing n_21 * 2 with the number 4.
Sspec.subs({Nspec[0, 0]: 3}).replace(sigma, lambda x: 2 * x).replace(Nspec[2, 1] * 2, 4)




# %% codecell
lambd = Function("lambda")
lambd_ = lambda matrix : sum(matrix)


vv(X, W)
# %% codecell
vv(A, B)
# %% codecell
L = lambd(S); L
# %% codecell
n(B,A).replace(n, vv)
# %% codecell
M = vv(X, W); M
# %% codecell
L.replace(n, vv).replace(sigmaApply, sigmaApply_)
# %% codecell
L.replace(n, v)
# %% codecell

L.replace(n, v).replace(sigmaApply, sigmaApply_)

temp = lambd(sigmaApply(M)); temp
temp.replace(sigmaApply, sigmaApply_)

# %% codecell
L.subs({A:X, B:W}).replace(n, vv).replace(sigmaApply, sigmaApply_)
# %% codecell
L.subs({n: v, A:X, B:W}).replace(sigmaApply, sigmaApply_).subs({Nspec[0, 1]: 34})

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
sigma(n_ij).diff(n_ij).subs({n_ij : Nspec[0, 0]})

# %% codecell
# ### WAY 2:
n_11 = Function('n_11')(Nspec[0, 0]); n_11

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
