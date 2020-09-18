# %% codecell
from sympy import Matrix, Symbol, derive_by_array, Lambda, Function, MatrixSymbol, Derivative
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






# %% markdown
# Defining $N = \nu(X, W) = X \times W$
#
# * $\nu : \mathbb{R}^{(n \times m) \times (m \times p)} \rightarrow \mathbb{R}^{n \times p}$
# * $N \in \mathbb{R}^{n \times p}$
# %% codecell
v = lambda a,b: a*b

vL = Lambda((a,b), a*b)

n = Function('v') #, Lambda((a,b), a*b))

vN = lambda mat1, mat2: Matrix(mat1.shape[0], mat2.shape[1], lambda i, j: Symbol("n_{}{}".format(i+1, j+1))); vN


Nelem = vN(X, W)
Nelem

# %% codecell
Nspec = v(X,W)
Nspec
# %% codecell
#N = v(X, W); N
N = n(A,B)
N









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
# way 2 of declaring S (better way)
sigma = Function('sigma')

sigmaApply = Function("sigma_apply") #lambda matrix:  matrix.applyfunc(sigma)

sigmaApply_ = lambda matrix: matrix.applyfunc(sigma)

S = sigmaApply(N); S

# %% codecell
Sspec = S.subs({A:X, B:W}).replace(n, v).replace(sigmaApply, sigmaApply_)
Sspec
# %% codecell
Selem = S.replace(n, vN).replace(sigmaApply, sigmaApply_)
Selem


# %% codecell
import itertools

elemToSpecD = dict(itertools.chain(*[[(Nelem[i, j], Nspec[i, j]) for j in range(2)] for i in range(3)]))

elemToSpec = list(elemToSpecD.items())

Matrix(elemToSpec)
# %% codecell
elemToSpecFuncD = dict(itertools.chain(*[[(Nelem[i, j], Function("n_{}{}".format(i + 1, j + 1))(Nspec[i, j])) for j in range(2)] for i in range(3)]))

elemToSpecFunc = list(elemToSpecFuncD.items())

Matrix(elemToSpecFunc)
# %% codecell
elemToSpecFuncArgsD = dict(itertools.chain(*[[(Nelem[i, j], Function("n_{}{}".format(i + 1, j + 1))(*X,*W)) for j in range(2)] for i in range(3)]))

elemToSpecFuncArgs = list(elemToSpecFuncArgsD.items())

Matrix(elemToSpecFuncArgs)
# %% codecell
Selem
# %% codecell
Selem.subs(elemToSpecD)
# %% codecell
Selem[0,1].diff(Nelem[0,1])
# %% codecell
Selem[0,1].diff(Nelem[0,1]).subs({Nelem[0,1] : Nspec[0,1]})
#Selem[0,1].diff(Nelem[0,1]).subs(dict([{Nelem[0,1] : Nspec[0,1]}]))

# %% codecell
Selem[0,1].diff(Nelem[0,1]).subs({Nelem[0,1] : Nspec[0,1]}).subs({Nspec[0,1] : 23})
# %% codecell
Selem[0,1].diff(Nelem[0,1]).subs({Nelem[0,1] : Nspec[0,1]}).replace(sigma, lambda x: 8*x**3)
# %% codecell
Selem[0,1].diff(Nelem[0,1]).replace(sigma, lambda x: 8*x**3)
# %% codecell
Selem[0,1].diff(Nelem[0,1]).replace(sigma, lambda x: 8*x**3).doit()
# %% codecell
# ### GOT IT: can replace now with expression and do derivative with respect to that expression.
Selem[0,1].diff(Nelem[0,1]).subs({Nelem[0,1] : Nspec[0,1]}).replace(sigma, lambda x: 8*x**3).doit()
# %% codecell
Selem[0,1].subs({Nelem[0,1] : Nspec[0,1]}).diff(X[0,1])#.subs({Nelem[0,1] : Nspec[0,1]})
# %% codecell
Selem

# %% codecell
nt = Nelem.subs(elemToSpecFunc); nt
# %% codecell
st = Selem.subs(elemToSpecFunc); st
# %% codecell
st.diff(nt)
# %% codecell
st[0,0].diff(st[0,0].args[0])
# %% codecell
temp = st[0,0].diff(X[0,0]); temp

#nt[0,0]

#temp.replace(Function("n_11")(nt[0,0].args[0]), nt[0,0].args[0])

#temp.subs({nt[0,0] : nt[0,0].args[0]})



# %% codecell
st[0,0].diff(st[1,0].args[0])
# %% codecell
Selem.diff(Nelem)
# %% codecell
Selem.diff(Nelem).subs(elemToSpecFunc)

# %% codecell
# CAN even replace elements after have done an operation on them!!! replacing n_21 * 2 with the number 4.
Sspec.subs({Nspec[0, 0]: 3}).replace(sigma, lambda x: 2 * x).replace(Nspec[2, 1] * 2, 4)









# %% codecell
lambd = Function("lambda")
lambd_ = lambda matrix : sum(matrix)

L = lambd(S); L
# %% codecell
L.replace(n, vN).replace(sigmaApply, sigmaApply_)
# %% codecell
#L.replace(n, vN).replace(sigmaApply, sigmaApply_).diff(Nelem[0,0])

# %% codecell
Lsum = L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)
Lsum
# %% codecell
Lsum.diff(Nelem)
# %% codecell
Lsum.subs(elemToSpec)#.diff(X[2,1])
# %% codecell
Lsum.subs(elemToSpecD).diff(X)
# %% codecell
specToElemD = {v:k for k,v in elemToSpecD.items()}
Lsum.subs(elemToSpecD).diff(X).subs(specToElemD)



# %% codecell
#### NOW DOING THE MATRIX SYMBOL DIFF EXPRESSION
Selem
# %% codecell
L

# %% codecell
from sympy import diff
# ### WARNING: this only works when size(X) == size(Y) else since size(W) != size(X) cannot subst B with W, so this operation won't work in my case.

Y = Matrix(3,3, lambda i,j: Symbol("y_{}{}".format(i+1,j+1)))

#diff(L.replace(A,A.T), A) # ERROR max recursion depth exceeded
#2+2
Lmat = L.subs({A:X, B:Y}).diff(X).replace(X, A).replace(Y,B); Lmat
# %% codecell
elem = Lmat[0,0][0,0];elem
# %% codecell
#Lmat.replace(n, vL) # error can't calc deriv .w.r.t to x11*w11 +...
# Lmat.replace(n, v) # error can't calc deriv .w.r.t to x11*w11 +...
elem.subs(n, vL)

# %% codecell
#elem.replace(n, v) # error cannot deriv wrt to X*W
# %% codecell
Selem
# %% codecell
# use replace n with vN instead of subs n with vL to get less specific output so it is easier to see since vL returns the xww*w11 +.... expressions
elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_)
# %% codecell
# Making matrix symbols again
Ss = MatrixSymbol('S', 3,2) #n by p
Ns = MatrixSymbol('N', 3,2) #n by p



short = elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_).replace(X,A).replace(Nelem, Ns).replace(Selem ,Ss)
short
# %% codecell
# Now going back to matrix form just to apply the last function LAMBDA
elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)
# %% codecell
# Making each of the n_ijs a function
#elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD)

Matrix(elemToSpecFuncArgs)
# %% codecell

long = elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecFuncArgsD)
long
# %% codecell
# short version again:
short
# %% codecell
# long.doit() # error as base exp thing
# %% codecell
# Trying step by step replacement approach:
elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).replace(Nelem, Ns).replace(X,A)
# %% codecell
# Seeing if replacing the order of replacing Ns matrix with Xs matrix makes a difference: ...
step = elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).replace(Nelem, Ns).replace(X,A).doit()
step
# %% codecell
elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).replace(X,A).replace(Nelem, Ns).doit()

# %% codecell
step.replace(Ns, Nelem)
# %% codecell
#step.replace(Ns, Nelem).replace(A,X).doit()#error immutable dense array has no attribute as base exp ...
elem2 = step[0,0].replace(Ns, Nelem)
elem2.replace(A,X).subs(elemToSpecFuncArgsD)

# %% codecell
#elem2.replace(A,X).subs(elemToSpecFuncArgsD).doit()
F = Nelem.subs(elemToSpecFuncArgsD); F
# %% codecell
F[0,0].diff(X[0,0])


# %% codecell
F[0,0].diff(X)
# %% codecell
F.diff(X)
# %% codecell
argsToSpecD = dict(zip(elemToSpecFuncArgsD.values(), elemToSpecD.values()))
argsToSpec = list(argsToSpecD.items())
Matrix(argsToSpec)
# %% codecell
F[0,0].diff(X[0,0]).subs(argsToSpecD)#.subs({elemToSpecFuncArgs[0][1] : Nspec[0,0]})
# %% codecell
F[0,0].diff(X[0,0]).subs(argsToSpecD).doit()
# %% codecell
# NOTE: using diff did not work, said immutable dense array cannot be subs-ed
derive_by_array(F, X).subs(argsToSpecD)

# %% codecell
derive_by_array(F, X).subs(argsToSpecD).doit()
# %% codecell
derive_by_array(F, W).subs(argsToSpecD).doit()




# %% codecell
elem2
# %% codecell

funcMat = elem2.subs(elemToSpecFuncArgsD).replace(A,X)#.diff(X)
funcMat

# %% codecell
#funcMat.doit() # error
#derive_by_array(funcMat, X)
# %% codecell
funcMat = elem2.subs(elemToSpecFuncD).replace(A,X)#.diff(X)
funcMat
# %% codecell
#funcMat.doit() # same error
#elem2.subs(elemToSpecFuncD).doit() # error
elem2

# %% codecell
# elem2.replace(A,X).doit() # error
# %% codecell
#elem2.replace(A,a).doit()#.subs(elemToSpecFuncArgsD).doit()
# ERROR everywhere what next todo? this approach worked before, where I make w.r.t. thing a real matrix, and leave the others a symbol so why isn't it working now?
# %% codecell
#elem2.replace(A,X).subs(elemToSpecFuncD).doit()
# ERROR this has to work though! Then can simply replace n_ijs with lambda


# %% codecell
elemToMatArgD = dict(itertools.chain(*[[(Nelem[i, j], Function("n_{}{}".format(i+1,j+1))(A,B) ) for j in range(2)] for i in range(3)]))

elemToMatArg = list(elemToMatArgD.items())

Matrix(elemToMatArg)

# %% codecell
matargToSpecD = dict(zip(elemToMatArgD.values(), elemToSpecD.values()))

matargToSpec = list(matargToSpecD.items())

Matrix(matargToSpec)

# %% codecell
#elem2.subs(elemToMatArgD).doit()#ERROR max recursion depth exceeeded
