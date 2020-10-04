```python title="codecell"
from sympy import Matrix, Symbol, derive_by_array, Lambda, Function, MatrixSymbol, Derivative, symbols, diff
from sympy import var
from sympy.abc import x, i, j, a, b
```

```python title="codecell"
def myvar(letter: str, i: int, j: int) -> Symbol:
    letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
    return letter_ij


n,m,p = 3,3,2

X = Matrix(n, m, lambda i,j : myvar('x', i, j)); X
```
```python title="codecell"
W = Matrix(m, p, lambda i,j : myvar('w', i, j)); W
```
```python title="codecell"
A = MatrixSymbol('X',3,3); Matrix(A)
B = MatrixSymbol('W',3,2)




```

```python title="codecell"
v = lambda a,b: a*b

vL = Lambda((a,b), a*b)

vL2 = Lambda((A,B), A*B)

n = Function('v') #, Lambda((a,b), a*b))

vN = lambda mat1, mat2: Matrix(mat1.shape[0], mat2.shape[1], lambda i, j: Symbol("n_{}{}".format(i+1, j+1))); vN


Nelem = vN(X, W)
Nelem
```

```python title="codecell"
Nspec = v(X,W)
Nspec
```
```python title="codecell"
#N = v(X, W); N
N = n(A,B)
N




```

```python title="codecell"
# way 2 of declaring S (better way)
sigma = Function('sigma')

sigmaApply = Function("sigma_apply") #lambda matrix:  matrix.applyfunc(sigma)

sigmaApply_ = lambda matrix: matrix.applyfunc(sigma)


S = sigmaApply(N); S
```

```python title="codecell"
Sspec = S.subs({A:X, B:W}).replace(n, v).replace(sigmaApply, sigmaApply_)
Sspec
```
```python title="codecell"
Selem = S.replace(n, vN).replace(sigmaApply, sigmaApply_)
Selem
```


```python title="codecell"
import itertools

elemToSpecD = dict(itertools.chain(*[[(Nelem[i, j], Nspec[i, j]) for j in range(2)] for i in range(3)]))

elemToSpec = list(elemToSpecD.items())

Matrix(elemToSpec)
```
```python title="codecell"
specToElemD = {v:k for k,v in elemToSpecD.items()}
specToElem = list(specToElemD.items())
Matrix(specToElem)
```
```python title="codecell"
elemToSpecFuncD = dict(itertools.chain(*[[(Nelem[i, j], Function("n_{}{}".format(i + 1, j + 1))(Nspec[i, j])) for j in range(2)] for i in range(3)]))

elemToSpecFunc = list(elemToSpecFuncD.items())

Matrix(elemToSpecFunc)
```
```python title="codecell"
elemToSpecFuncArgsD = dict(itertools.chain(*[[(Nelem[i, j], Function("n_{}{}".format(i + 1, j + 1))(*X,*W)) for j in range(2)] for i in range(3)]))

elemToSpecFuncArgs = list(elemToSpecFuncArgsD.items())

Matrix(elemToSpecFuncArgs)
```
```python title="codecell"
elemToMatArgD = dict(itertools.chain(*[[(Nelem[i, j], Function("n_{}{}".format(i+1,j+1))(A,B) ) for j in range(2)] for i in range(3)]))

elemToMatArg = list(elemToMatArgD.items())

Matrix(elemToMatArg)
```

```python title="codecell"
matargToSpecD = dict(zip(elemToMatArgD.values(), elemToSpecD.values()))

matargToSpec = list(matargToSpecD.items())

Matrix(matargToSpec)
```


```python title="codecell"
Selem
```
```python title="codecell"
Sspec = Selem.subs(elemToSpecD)
Sspec
```

```python title="codecell"
# CAN even replace elements after have done an operation on them!!! replacing n_21 * 2 with the number 4.
Sspec.subs({Nspec[0, 0]: 3}).replace(sigma, lambda x: 2 * x).replace(Nspec[2, 1] * 2, 4)








```

```python title="codecell"
lambd = Function("lambda")
lambd_ = lambda matrix : sum(matrix)

i, j = symbols('i j')
M = MatrixSymbol('M', i, j)# abstract shape
#sigmaApply_L = Lambda(M, M.applyfunc(sigma))

lambda_L = Lambda(M, sum(M))

n = Function("nu",applyfunc=True)

L = lambd(sigmaApply(n(A,B))); L
```
```python title="codecell"
L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B)
```
```python title="codecell"
L.replace(n,vN).replace(sigmaApply, sigmaApply_)
```
```python title="codecell"
L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)
```
```python title="codecell"
L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD)
```

```python title="codecell"
L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(W).subs(specToElemD)
```


```python title="codecell"
# Now verifying the above rule by applying the composition thing piece by piece via multiplication:
L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(B)
```
```python title="codecell"
L.replace(n,vL).replace(sigmaApply,sigmaApply_)#.diff()
```
```python title="codecell"
L.replace(n,vL)
```
```python title="codecell"
L.replace(n,vL).diff(sigmaApply(A*B))
```
```python title="codecell"
#L.replace(n,vL).diff(sigmaApply(A*B)).replace(sigmaApply,sigmaApply_)
## ERROR cannot do that
```
```python title="codecell"
sigmaApply_L = Lambda(M, M.applyfunc(sigma))
#L.replace(n,vL).diff(sigmaApply(A*B)).subs(sigmaApply,sigmaApply_L) ## ERROR

nL = Lambda((A,B), n(A,B)); nL
```


```python title="codecell"
# Try to create function cpmpositions :
from functools import reduce

def compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))

def compose(*fs):
    return reduce(compose2, fs)
```

```python title="codecell"
f, g, h = symbols("f g h ", cls=Function)
diff(f(g(h(x))), x)
```
```python title="codecell"
compose(f,g,h)(x)
```
```python title="codecell"
compose(f,g,h)
```
```python title="codecell"
compose(lambd, sigmaApply, n)(A,B)
```
```python title="codecell"
#diff(compose(lambd, sigmaApply, n)(A.T,B), A)


#compose(lambd, sigmaApply, n)(A,B).diff(lambd)
diff(compose(lambd, sigmaApply, n)(A,B), compose(lambd, sigmaApply, n)(A,B))
```

```python title="codecell"
d = diff(compose(lambd, sigmaApply, n)(A,B), n(A,B)); d
```
```python title="codecell"
#compose(lambd, sigmaApply, n)(A,B).diff(A) # ERROR
Lc = compose(lambd, sigmaApply, n)(A, B)
Lc.replace(n, vL).replace(sigmaApply, sigmaApply_)
```
```python title="codecell"
# Same result as replacing in L
#Lc.replace(n, vL).replace(sigmaApply, sigmaApply_).diff(B)
Lc.replace(n, v).replace(sigmaApply, sigmaApply_).diff(B)
```

```python title="codecell"

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
```
```python title="codecell"
sigmaApply_L(A*B).diff(B)
```
```python title="codecell"
sigmaApply_L(A).diff(A).subs(A,X).doit()
```
```python title="codecell"

```

```python title="codecell"
sigmaApply_L(A*B).diff(B).subs(A,X).subs(B,W)#.doit()
### CANNOT go farther here because of noncommutative scalars in matrix error
```
```python title="codecell"
#sigmaApply_L(X*W).subs(specToElemD).subs(elemToMatArgD).diff(B)#
```
```python title="codecell"
sigmaApply_L(A*B).diff(B)#.replace(A,X).replace(B,W).replace(X*W,Nelem)
```

```python title="codecell"
sigmaApply_L(A*B).diff(B).subs({A*B : vN(A,B)})
```
```python title="codecell"
sigmaApply_L(A*B).diff(B).subs({A*B : vN(A,B)}).doit()
```
```python title="codecell"
sigmaApply_L(A*B).diff(B).subs({A*B : vN(A,B)}).subs(elemToMatArgD)
```
```python title="codecell"
sigmaApply_L(A*B).diff(B).subs({A*B : vN(A,B)}).subs(elemToMatArgD).doit()
```
```python title="codecell"
sigmaApply_L(A*B).diff(B).subs({A*B : vN(A,B)}).doit()
```
```python title="codecell"
#part1=sigmaApply_L(A*B).diff(B).subs({A*B : vN(A,B)}).doit()
```
```python title="codecell"
part1 = compose(sigmaApply, n)(A,B).replace(n, v).replace(sigmaApply, sigmaApply_).diff(B).subs({A*B : vN(A,B)}).doit()


part1.subs({A:X}) # replace won't work here
```
```python title="codecell"
part1.subs({A:X}).doit()
```
<!-- #region markdown -->
### COMPARE: Symbolic form vs. Direct form vs. Step by Step form (which goes from symbolic to direct form by replacing)
#### Symbolic Abstract Form (with respect to W):
<!-- #endregion -->
```python title="codecell"
Lc = compose(lambd, sigmaApply, n)(A, B)
symb = Lc.replace(n, v).replace(sigmaApply, sigmaApply_).diff(B)
symb
```
<!-- #region markdown -->
#### Direct form: (after the symbolic form)
<!-- #endregion -->
```python title="codecell"
Lc.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(W).subs(specToElemD)
```
<!-- #region markdown -->
Just placing the "n" values right in place of the "epsilons" using the "doit" function:
<!-- #endregion -->
```python title="codecell"
direct = Lc.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(W).subs(specToElemD).doit()
direct
```
<!-- #region markdown -->
#### Step by Step Form:
<!-- #endregion -->
```python title="codecell"
assert symb == Lc.replace(n, v).replace(sigmaApply, sigmaApply_).diff(B)

symb.subs({A*B : vN(A,B)})#.doit()
```
```python title="codecell"
symb.subs({A*B : vN(A,B)}).doit() # the dummy variable for lambda still stays unapplied
```
```python title="codecell"
symb.subs({A*B : vN(A,B)}).subs({A:X}).doit() # two doits are equivalent to the last one at the end
```
```python title="codecell"
# Creating a special symbol just for the lambda_L function that has the appropriate shape after multiplying A*B. Supposed to represent a matrix R s.t. R == A * B (so that the indices after applying lambda_L are correct)
ABres = MatrixSymbol("R", A.shape[0], B.shape[1])
lambd_L = Lambda(ABres, sum(ABres))


symb.subs({A*B : vN(A,B)}).subs({A:X}).doit()#subs({lambd: lambd_L}).doit()
```

```python title="codecell"
# yay finall got the dummy variable in the lambd to be applied!!
unapplied = sigmaApply_L(vN(A,B))
applied = unapplied.doit()

symb.subs({A*B : vN(A,B)}).subs({A:X}).doit().replace(unapplied, applied)
```

```python title="codecell"
# THIS SEEMS WRONG : ??? how to tell for sure?
lambd(Selem).diff(Selem).replace(lambd, lambd_L).doit()
```
```python title="codecell"
# THis seems right:
dL_dS = lambd(Selem).replace(lambd, lambd_L).diff(Selem)
dL_dS
```
```python title="codecell"
lambd(Selem)
```
```python title="codecell"
# This is the X^T * dS/dN part
compose(sigmaApply, n)(A,B).replace(n, v).replace(sigmaApply, sigmaApply_).diff(B).subs({A*B : vN(A,B)}).doit()
```
```python title="codecell"
N = MatrixSymbol("N", A.shape[0], B.shape[1])
Matrix(N)
```
```python title="codecell"
dS_dN = compose(sigmaApply)(N).replace(sigmaApply, sigmaApply_).diff(N).subs({N : vN(A,B)}).doit()
# WRONG:
#dS_dN = sigmaApply(Nelem).replace(sigmaApply, sigmaApply_).diff(Matrix(Nelem))
dS_dN
```

```python title="codecell"
from sympy.physics.quantum import TensorProduct

#TensorProduct( X.T, dS_dN)


dN_dW = X.T
dS_dW = dN_dW * dS_dN
dS_dW
#HadamardProduct(X.T, dS_dN)
```
```python title="codecell"
from sympy import HadamardProduct

dL_dW = HadamardProduct(dS_dW , dL_dS)
dL_dW
```
<!-- #region markdown -->
One more time as complete symbolic form:
$$
\begin{aligned}
\frac{\partial L}{\partial W} &= \frac{\partial N}{\partial W} \times \frac{\partial S}{\partial N} \odot \frac{\partial L}{\partial S} \\
&= X^T \times  \frac{\partial S}{\partial N} \odot \frac{\partial L}{\partial S}
\end{aligned}
$$
where $\odot$ signifies the Hadamard product and $\times$ is matrix multiplication.
<!-- #endregion -->
```python title="codecell"
HadamardProduct(dN_dW * dS_dN, dL_dS)
```
```python title="codecell"
direct
```
```python title="codecell"
assert HadamardProduct(dN_dW * dS_dN, dL_dS).equals(direct)
```
```python title="codecell"
# too long to see in editor:
symb.subs({A*B : vN(A,B)}).subs({A:X}).doit().replace(lambd, lambd_L)
```
```python title="codecell"
symb.subs({lambd: lambd_L})
```
```python title="codecell"
print(symb.subs({lambd: lambd_L}))
```


```python title="codecell"
LcL = compose(lambd_L, sigmaApply, n)(A, B)
LcL
```
```python title="codecell"
symbL = LcL.replace(n, v).replace(sigmaApply, sigmaApply_)#.diff(A)
symbL
```
```python title="codecell"
compose(lambd, sigmaApply, n)(A,B).replace(n,v).subs({lambd:lambd_L})#.subs({sigmaApply : sigmaApply_L})
```
```python title="codecell"
compose(lambd, sigmaApply, n)(A,B).replace(n,v).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_L)
```
```python title="codecell"
compose(lambd, sigmaApply, n)(A,B).replace(n,v).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_L).doit()
```
```python title="codecell"
compose(lambd, sigmaApply, n)(A,B).replace(n,v).diff(B).doit()#replace(sigmaApply, sigmaApply_)#.replace(lambd, lambd_L).diff(B)
```
```python title="codecell"
compose(lambd, sigmaApply, n)(A,B).replace(n,v).diff(B).replace(lambd, lambd_L)
```
```python title="codecell"
compose(lambd, sigmaApply, n)(A,B).replace(lambd, lambd_L)
```
```python title="codecell"
compose(lambd, sigmaApply, n)(A,B).replace(lambd, lambd_L).replace(n, v).replace(sigmaApply, sigmaApply_)
```
```python title="codecell"
compose(lambd, sigmaApply, n)(A,B).replace(lambd, lambd_L).replace(n, v).replace(sigmaApply, sigmaApply_).doit()#.diff(B)
```
```python title="codecell"
compose(lambd, sigmaApply, n)(A,B).replace(lambd, lambd_L).replace(n, v).replace(sigmaApply, sigmaApply_).doit().diff(Matrix(B)).doit()
```

```python title="codecell"
sigmaApply = Function("sigma_apply", subscriptable=True)

#compose(lambd, sigmaApply, n)(A,B).replace(n,v).diff(B).replace(lambd, lambd_L).doit()# ERROR sigma apply is not subscriptable

#compose(lambd, sigmaApply, n)(A,B).replace(n,v).diff(B).subs({sigmaApply: sigmaApply_L})








```

```python title="codecell"
compose(sigmaApply_L, sigmaApply_L)(A)
```
```python title="codecell"
x = Symbol('x', applyfunc=True)
#compose(sigmaApply_, sigmaApply_)(x)##ERROR
compose(sigmaApply_, sigmaApply_)(A)#.replace(A,f(x))
```
```python title="codecell"
compose(lambda_L, nL)(A,B)
```
```python title="codecell"
n = Function("v", subscriptable=True) # doesn't work for below
#compose(lambda_L, n)(A,B).doit()
```

```python title="codecell"
VL = Lambda((A,B), Lambda((A,B), MatrixSymbol("V", A.shape[0], B.shape[1])))
VL
```
```python title="codecell"
VL(A,B)
```
```python title="codecell"
#saL = Lambda(A, Lambda(A, sigma(A)))
saL = Lambda(x, Lambda(x, sigma(x)))
#saL(n(A,B))## ERROR : the ultimate test failed: cannot even make this take an arbitrary function
#saL(n)
#s = lambda x : Lambda(x, sigma(x))
s = lambda x : sigma(x)
s(n(A,B))
```
```python title="codecell"
#sL = lambda x : Lambda(x, sigma(x))
#sL = Lambda(x, lambda x: sigma(x))

sL = Lambda(x, Lambda(x, sigma(x)))
sL
#sL(A)
```
