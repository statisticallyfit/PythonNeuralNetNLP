# %% codecell
from sympy import Matrix, Symbol, derive_by_array, Lambda, Function, MatrixSymbol, Derivative, symbols, diff
from sympy import var
from sympy.abc import x, i, j, a, b


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
v = lambda a,b: a*b

vL = Lambda((a,b), a*b)

vL2 = Lambda((A,B), A*B)

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
specToElemD = {v:k for k,v in elemToSpecD.items()}
specToElem = list(specToElemD.items())
Matrix(specToElem)
# %% codecell
elemToSpecFuncD = dict(itertools.chain(*[[(Nelem[i, j], Function("n_{}{}".format(i + 1, j + 1))(Nspec[i, j])) for j in range(2)] for i in range(3)]))

elemToSpecFunc = list(elemToSpecFuncD.items())

Matrix(elemToSpecFunc)
# %% codecell
elemToSpecFuncArgsD = dict(itertools.chain(*[[(Nelem[i, j], Function("n_{}{}".format(i + 1, j + 1))(*X,*W)) for j in range(2)] for i in range(3)]))

elemToSpecFuncArgs = list(elemToSpecFuncArgsD.items())

Matrix(elemToSpecFuncArgs)
# %% codecell
elemToMatArgD = dict(itertools.chain(*[[(Nelem[i, j], Function("n_{}{}".format(i+1,j+1))(A,B) ) for j in range(2)] for i in range(3)]))

elemToMatArg = list(elemToMatArgD.items())

Matrix(elemToMatArg)

# %% codecell
matargToSpecD = dict(zip(elemToMatArgD.values(), elemToSpecD.values()))

matargToSpec = list(matargToSpecD.items())

Matrix(matargToSpec)


# %% codecell
Selem
# %% codecell
Sspec = Selem.subs(elemToSpecD)
Sspec

# %% codecell
# CAN even replace elements after have done an operation on them!!! replacing n_21 * 2 with the number 4.
Sspec.subs({Nspec[0, 0]: 3}).replace(sigma, lambda x: 2 * x).replace(Nspec[2, 1] * 2, 4)









# %% codecell
lambd = Function("lambda")
lambd_ = lambda matrix : sum(matrix)

i, j = symbols('i j')
M = MatrixSymbol('M', i, j)# abstract shape
#sigmaApply_L = Lambda(M, M.applyfunc(sigma))

lambda_L = Lambda(M, sum(M))

n = Function("nu",applyfunc=True)

L = lambd(sigmaApply(n(A,B))); L
# %% codecell
L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B)
# %% codecell
L.replace(n,vN).replace(sigmaApply, sigmaApply_)
# %% codecell
L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)
# %% codecell
L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD)

# %% codecell
L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(W).subs(specToElemD)


# %% codecell
# Now verifying the above rule by applying the composition thing piece by piece via multiplication:
L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(B)
# %% codecell
L.replace(n,vL).replace(sigmaApply,sigmaApply_)#.diff()
# %% codecell
L.replace(n,vL)
# %% codecell
L.replace(n,vL).diff(sigmaApply(A*B))
# %% codecell
#L.replace(n,vL).diff(sigmaApply(A*B)).replace(sigmaApply,sigmaApply_)
## ERROR cannot do that
# %% codecell
sigmaApply_L = Lambda(M, M.applyfunc(sigma))
#L.replace(n,vL).diff(sigmaApply(A*B)).subs(sigmaApply,sigmaApply_L) ## ERROR

nL = Lambda((A,B), n(A,B)); nL


# %% codecell
# Try to create function cpmpositions :
from functools import reduce

def compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))

def compose(*fs):
    return reduce(compose2, fs)

# %% codecell
f, g, h = symbols("f g h ", cls=Function)
diff(f(g(h(x))), x)
# %% codecell
compose(f,g,h)(x)
# %% codecell
compose(f,g,h)
# %% codecell
compose(lambd, sigmaApply, n)(A,B)
# %% codecell
#diff(compose(lambd, sigmaApply, n)(A.T,B), A)


#compose(lambd, sigmaApply, n)(A,B).diff(lambd)
diff(compose(lambd, sigmaApply, n)(A,B), compose(lambd, sigmaApply, n)(A,B))

# %% codecell
d = diff(compose(lambd, sigmaApply, n)(A,B), n(A,B)); d
# %% codecell
#compose(lambd, sigmaApply, n)(A,B).diff(A) # ERROR
Lc = compose(lambd, sigmaApply, n)(A, B)
Lc.replace(n, vL).replace(sigmaApply, sigmaApply_)
# %% codecell
# Same result as replacing in L
#Lc.replace(n, vL).replace(sigmaApply, sigmaApply_).diff(B)
Lc.replace(n, v).replace(sigmaApply, sigmaApply_).diff(B)

# %% codecell

funcToMat = lambda func: Matrix([func])
funcToMat(f)
A.applyfunc(sigma)
#funcToMat(f).applyfunc(sigma)
from sympy import FunctionMatrix
F = MatrixSymbol("F",3,3)#(FunctionMatrix(3,3, f))
FL = Lambda(F, n(A,B))
FL
gL = lambda A: A.applyfunc(sigma)
gL(A)
temp = lambda n : n
temp(n(A,B))
#sigmaApply_L(A).subs(A, Lambda((A,B), vL(A,B)))# arg must be matrix instance
# %% codecell
sigmaApply_L(A*B).diff(B)
# %% codecell
sigmaApply_L(A).diff(A).subs(A,X).doit()
# %% codecell

# %% codecell
sigmaApply_L(A*B).diff(B).subs(A,X).subs(B,W)#.doit()
### CANNOT go farther here because of noncommutative scalars in matrix error
# %% codecell
#sigmaApply_L(X*W).subs(specToElemD).subs(elemToMatArgD).diff(B)#
# %% codecell
sigmaApply_L(A*B).diff(B)#.replace(A,X).replace(B,W).replace(X*W,Nelem)

# %% codecell
sigmaApply_L(A*B).diff(B).subs({A*B : vN(A,B)})
# %% codecell
sigmaApply_L(A*B).diff(B).subs({A*B : vN(A,B)}).doit()
# %% codecell
sigmaApply_L(A*B).diff(B).subs({A*B : vN(A,B)}).subs(elemToMatArgD)
# %% codecell
sigmaApply_L(A*B).diff(B).subs({A*B : vN(A,B)}).subs(elemToMatArgD).doit()
# %% codecell
sigmaApply_L(A*B).diff(B).subs({A*B : vN(A,B)}).doit()
# %% codecell
#part1=sigmaApply_L(A*B).diff(B).subs({A*B : vN(A,B)}).doit()
# %% codecell
part1 = compose(sigmaApply, n)(A,B).replace(n, v).replace(sigmaApply, sigmaApply_).diff(B).subs({A*B : vN(A,B)}).doit()


part1.subs({A:X}) # replace won't work here
# %% codecell
part1.subs({A:X}).doit()
# %% markdown [markdown]
# ### COMPARE: Symbolic form vs. Direct form vs. Step by Step form (which goes from symbolic to direct form by replacing)
# #### Symbolic Abstract Form (with respect to W):
# %% codecell
Lc = compose(lambd, sigmaApply, n)(A, B)
symb = Lc.replace(n, v).replace(sigmaApply, sigmaApply_).diff(B)
symb
# %% markdown [markdown]
# #### Direct form: (after the symbolic form)
# %% codecell
Lc.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(W).subs(specToElemD)
# %% markdown [markdown]
# Just placing the "n" values right in place of the "epsilons" using the "doit" function:
# %% codecell
direct = Lc.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(W).subs(specToElemD).doit()
direct
# %% markdown [markdown]
# #### Step by Step Form:
# %% codecell
assert symb == Lc.replace(n, v).replace(sigmaApply, sigmaApply_).diff(B)

symb.subs({A*B : vN(A,B)})#.doit()
# %% codecell
symb.subs({A*B : vN(A,B)}).doit() # the dummy variable for lambda still stays unapplied
# %% codecell
symb.subs({A*B : vN(A,B)}).subs({A:X}).doit() # two doits are equivalent to the last one at the end
# %% codecell
# Creating a special symbol just for the lambda_L function that has the appropriate shape after multiplying A*B. Supposed to represent a matrix R s.t. R == A * B (so that the indices after applying lambda_L are correct)
ABres = MatrixSymbol("R", A.shape[0], B.shape[1])
lambd_L = Lambda(ABres, sum(ABres))


symb.subs({A*B : vN(A,B)}).subs({A:X}).doit()#subs({lambd: lambd_L}).doit()

# %% codecell
# yay finall got the dummy variable in the lambd to be applied!!
unapplied = sigmaApply_L(vN(A,B))
applied = unapplied.doit()

symb.subs({A*B : vN(A,B)}).subs({A:X}).doit().replace(unapplied, applied)

# %% codecell
# THIS SEEMS WRONG : ??? how to tell for sure?
lambd(Selem).diff(Selem).replace(lambd, lambd_L).doit()
# %% codecell
# THis seems right:
dL_dS = lambd(Selem).replace(lambd, lambd_L).diff(Selem)
dL_dS
# %% codecell
lambd(Selem)
# %% codecell
# This is the X^T * dS/dN part
compose(sigmaApply, n)(A,B).replace(n, v).replace(sigmaApply, sigmaApply_).diff(B).subs({A*B : vN(A,B)}).doit()
# %% codecell
N = MatrixSymbol("N", A.shape[0], B.shape[1])
Matrix(N)
# %% codecell
dS_dN = compose(sigmaApply)(N).replace(sigmaApply, sigmaApply_).diff(N).subs({N : vN(A,B)}).doit()
# WRONG:
#dS_dN = sigmaApply(Nelem).replace(sigmaApply, sigmaApply_).diff(Matrix(Nelem))
dS_dN

# %% codecell
from sympy.physics.quantum import TensorProduct

#TensorProduct( X.T, dS_dN)


dN_dW = X.T
dS_dW = dN_dW * dS_dN
dS_dW
#HadamardProduct(X.T, dS_dN)
# %% codecell
from sympy import HadamardProduct

dL_dW = HadamardProduct(dS_dW , dL_dS)
dL_dW
# %% markdown [markdown]
# One more time as complete symbolic form:
# $$
# \begin{aligned}
# \frac{\partial L}{\partial W} &= \frac{\partial N}{\partial W} \times \frac{\partial S}{\partial N} \odot \frac{\partial L}{\partial S} \\
# &= X^T \times  \frac{\partial S}{\partial N} \odot \frac{\partial L}{\partial S}
# \end{aligned}
# $$
# where $\odot$ signifies the Hadamard product and $\times$ is matrix multiplication.
# %% codecell
HadamardProduct(dN_dW * dS_dN, dL_dS)
# %% codecell
direct
# %% codecell
assert HadamardProduct(dN_dW * dS_dN, dL_dS).equals(direct)
# %% codecell
# too long to see in editor:
symb.subs({A*B : vN(A,B)}).subs({A:X}).doit().replace(lambd, lambd_L)
# %% codecell
symb.subs({lambd: lambd_L})
# %% codecell
print(symb.subs({lambd: lambd_L}))


# %% codecell
LcL = compose(lambd_L, sigmaApply, n)(A, B)
LcL
# %% codecell
symbL = LcL.replace(n, v).replace(sigmaApply, sigmaApply_)#.diff(A)
symbL
# %% codecell
compose(lambd, sigmaApply, n)(A,B).replace(n,v).subs({lambd:lambd_L})#.subs({sigmaApply : sigmaApply_L})
# %% codecell
compose(lambd, sigmaApply, n)(A,B).replace(n,v).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_L)
# %% codecell
compose(lambd, sigmaApply, n)(A,B).replace(n,v).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_L).doit()
# %% codecell
compose(lambd, sigmaApply, n)(A,B).replace(n,v).diff(B).doit()#replace(sigmaApply, sigmaApply_)#.replace(lambd, lambd_L).diff(B)
# %% codecell
compose(lambd, sigmaApply, n)(A,B).replace(n,v).diff(B).replace(lambd, lambd_L)
# %% codecell
compose(lambd, sigmaApply, n)(A,B).replace(lambd, lambd_L)
# %% codecell
compose(lambd, sigmaApply, n)(A,B).replace(lambd, lambd_L).replace(n, v).replace(sigmaApply, sigmaApply_)
# %% codecell
compose(lambd, sigmaApply, n)(A,B).replace(lambd, lambd_L).replace(n, v).replace(sigmaApply, sigmaApply_).doit()#.diff(B)
# %% codecell
compose(lambd, sigmaApply, n)(A,B).replace(lambd, lambd_L).replace(n, v).replace(sigmaApply, sigmaApply_).doit().diff(Matrix(B)).doit()

# %% codecell
sigmaApply = Function("sigma_apply", subscriptable=True)

#compose(lambd, sigmaApply, n)(A,B).replace(n,v).diff(B).replace(lambd, lambd_L).doit()# ERROR sigma apply is not subscriptable

#compose(lambd, sigmaApply, n)(A,B).replace(n,v).diff(B).subs({sigmaApply: sigmaApply_L})









# %% codecell
compose(sigmaApply_L, sigmaApply_L)(A)
# %% codecell
x = Symbol('x', applyfunc=True)
#compose(sigmaApply_, sigmaApply_)(x)##ERROR
compose(sigmaApply_, sigmaApply_)(A)#.replace(A,f(x))
# %% codecell
compose(lambda_L, nL)(A,B)
# %% codecell
n = Function("v", subscriptable=True) # doesn't work for below
#compose(lambda_L, n)(A,B).doit()

# %% codecell
VL = Lambda((A,B), Lambda((A,B), MatrixSymbol("V", A.shape[0], B.shape[1])))
VL
# %% codecell
VL(A,B)
# %% codecell
#saL = Lambda(A, Lambda(A, sigma(A)))
saL = Lambda(x, Lambda(x, sigma(x)))
#saL(n(A,B))## ERROR : the ultimate test failed: cannot even make this take an arbitrary function
#saL(n)
#s = lambda x : Lambda(x, sigma(x))
s = lambda x : sigma(x)
s(n(A,B))
# %% codecell
#sL = lambda x : Lambda(x, sigma(x))
#sL = Lambda(x, lambda x: sigma(x))

sL = Lambda(x, Lambda(x, sigma(x)))
sL
#sL(A)
