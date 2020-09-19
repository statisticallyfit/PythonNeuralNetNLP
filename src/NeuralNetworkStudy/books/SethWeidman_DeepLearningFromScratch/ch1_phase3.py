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


ns,ms,ps = 3,3,2

X = Matrix(ns, ms, lambda i,j : myvar('x', i, j)); X
# %% codecell
W = Matrix(ms, ps, lambda i,j : myvar('w', i, j)); W
# %% codecell
#TODO how to make matrix symbols commutative?
# A = MatrixSymbol('X',ns,ms, is_commutative=True); Matrix(A)
A = MatrixSymbol('X',ns,ms); Matrix(A)
B = MatrixSymbol('W',ms,ps)






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
Lsum.subs(elemToSpecD)#.diff(X[2,1])
# %% codecell
Lsum.subs(elemToSpecD).diff(X)
# %% codecell
# METHOD 1: direct matrix diff
#
### END RESULT ACHIEVED HERE (this is the end result and the most specific form of the result of the  matrix differentiation, when sigma is unknown)
specToElemD = {v:k for k,v in elemToSpecD.items()}

assert Lsum == L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)
Lsum.subs(elemToSpecD).diff(X).subs(specToElemD)



# %% codecell
# METHOD 2: doing matrix symbol diff
#
#### NOW DOING THE MATRIX SYMBOL DIFF EXPRESSION (trying to achieve a form that shows the chain rule w.r.t to matrix symbol)
Selem
# %% codecell
L
# %% codecell
#L.replace(A, A.T).diff(A) #ERROR: fatal python error ... why??

# %% codecell
#L.replace(n,v).diff(A).replace(sigmaApply, sigmaApply_) # ERROR
#L.replace(n,vN).subs(elemToSpecFuncD).replace(sigmaApply, sigmaApply_).diff(X) # why the zero matrix?

# %% codecell
L.replace(n,v).diff(A)
# %% codecell
L.replace(n,vL).diff(A)

# %% codecell
#L.replace(n,v).diff(A).replace(lambd,lambd_) ### ERROR sigma object is not iterable
#L.replace(n,vL).diff(A).replace(sigmaApply, sigmaApply_)### ERROR
#L.replace(n,v).diff(A).replace(sigmaApply, sigmaApply_) ### ERROR dummy object has no attribute applyfunc
# %% codecell
#L.replace(sigmaApply, sigmaApply_).diff(A) # ERROR
# L.replace(lambd, lambd_) # ERROR


### METHOD 0: (prepare by substituting n --> v, then sigmaApply --> sigma)
L.replace(n, v).replace(sigmaApply, sigmaApply_)#.replace(lambd, lambd_)
# %% codecell
### METHOD 0: the matrix diff rule in the most abstract form possible
L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(A)
# %% codecell
### SUCCESS! We see now that the matrix chain rule indeed makes the X transpose factor out on the left!!! (while compared to the above, the matrix transpose W^T factors out on the right, just like the book says (page 45 in the NOTE section of Seth Weidman book))
L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B)

# %% codecell
#L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B).replace(B,W).replace(A,X) # ## ERROR non commutative scalars in matrix
# L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B).replace(lambd, lambd_).replace(B,W).replace(A,X)# ## ERROR dummy object not iterable
# %% codecell

#L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(A).replace(lambd, lambd_) ### ERROR: dummy object not iterable (probably means when in the above expression we have epsilon = sigmaApply(XW) that we cannot iterate over this expression)
# %% codecell
# Replacing lambda first: BAD
#L.replace(n, v).replace(lambd, lambd_) ## ERROR sigma apply object not ieterable
# Replacing sigma first: BAD
# L.replace(sigmaApply, sigmaApply_)### ERROR v object has no attribute applyfunc
# %% codecell
# Replacing n first: GOOD (need to go from inner nesting to outermost function, never any other way)
L.replace(n, v).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)
# %% codecell
# ### END RESULT of METHOD 2:
L.replace(n, v).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).diff(Matrix(A))
# %% codecell
# COmpare the above matrix symbol way with the Lsum way:
#
# ### END RESULT of METHOD 1:
#Lsum = L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)

L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(X)#.subs(specToElemD)
# %% codecell


# %% codecell
## COMPARING METHOD 0 (abstract way) with METHOD 2 (direct way) when differentiating .w.r.t to X vs. w.r.t to W

### With respect to X (abstract)
L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(A)
# %% codecell
### With respect to X (direct)
L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(X).subs(specToElemD)
# %% codecell
### With respect to W (abstract)
L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(B)
# %% codecell
### With respect to W (direct)

L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(W).subs(specToElemD)
# %% codecell
# %% codecell
# %% codecell
# %% codecell
### NEXT: try to substitute the X, W matrices step by step to see if you can come to the same result as the direct forms above (from method 2 or 1)
from sympy import simplify, expand

#simplify(L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(B).subs({A:X, B:W})) ### ERROR max recursion depth exceeded
L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(B).subs({A:X, B:W})
# %% codecell
L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B)#.subs({A:X, B:W})


# %% codecell
L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B)
# %% codecell
#L.replace(n,v).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)

# %% codecell
#L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B).subs({A:X, B:W}).replace(lambd, lambd_) ### ERROR dummy object not iterable
L.replace(n,v).diff(A)
# %% codecell
#L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(A).replace(A,Matrix(A))##ERROR noncommutative matrix scalars
# WANT: to be able to do diff and have the expression come out as above with X^T on left and W^T on right, when using just this form, with abstract form v:
L.replace(A,A.T).replace(B,B.T)
# %% codecell
# Error if applying sigma to the v function because it sais v has no attribute applyfunc to trying now to making it have the attriute applyfunc.
y = Function('y', applyfunc=True, real=True)



from sympy.utilities.lambdify import lambdify, implemented_function
z = implemented_function('z', y)
lam_z = lambdify([A,B], z(A,B)); lam_z
# %% codecell
lam_z(A,B)
# %% codecell
Lz = lambd(sigmaApply(lam_z(A,B)))
Lz
# %% codecell
Ly = lambd(sigmaApply(y(A,B)))
Ly
# %% codecell

Ly.replace(A,A.T).replace(B,B.T)#.replace(sigmaApply, sigmaApply_)
# %% codecell
# TODO next step: to apply the sigma to get that applied functor expression but here get error saying bol object not callable ...??
Ly.replace(A,A.T).replace(B,B.T)#.replace(sigmaApply, sigmaApply_)

# %% codecell
# TODO always get fatal python error here, as if it can't deal with two matrix args!!
#Ly.replace(A,A.T).replace(B,B.T).diff(A)
def siga(mat: Matrix) -> Matrix:
     #lst = mat.tolist()
     nr, nc = mat.shape

     applied = [[sigma(mat[i,j]) for j in range(0, nc)] for i in range(0, nr)]

     return Matrix(applied)
sigmaApply_(Nelem)
siga(Nelem)
#siga2 = Lambda(a, siga(a))
# %% codecell
Ly.replace(A, A.T).replace(B, b).diff(b)#.replace(sigmaApply, siga)
# %% codecell
b = Symbol('b', commutative=True)
#Lz.replace(A,A.T).replace(B,b).diff(A)
# TODO still problem with bool object here even though the y-function is lambdified.
#sig = lambda mat: mat.applyfunc(lambda)
sigma.__dict__
#Lz.replace(sigmaApply, sigmaApply_)

# %% codecell
# Ly.replace(B, b).diff(A)#.replace(sigmaApply, siga)### ERROR noncommutative matrix scalars not supported
Ly.replace(A, A.T).replace(B, b).diff(b).replace(b, B).replace(A.T, A)#.replace(sigmaApply, siga)
# %% codecell
# NEXT: try to replace the sigma apply, not working
n.__dict__
# %% codecell
y.__dict__
# TODO HERE
#https://stackoverflow.com/questions/12614334/typeerror-bool-object-is-not-callable
# %% codecell
# %% codecell
# %% codecell
from sympy import diff
# ### WARNING: this only works when size(X) == size(Y) else since size(W) != size(X) cannot subst B with W, so this operation won't work in my case.

#X = Matrix(3,3, lambda i,j: Symbol("x_{}{}".format(i+1,j+1))); Matrix(X)
# Create another matrix instead of W so that it matches size of X during diff(X) operation, since otherwise the diff by X doesn't work, says X and W need to be same size.

Wtemp = Matrix(*X.shape, lambda i,j: Symbol("t_{}{}".format(i+1,j+1))); Matrix(Wtemp)
# %% codecell
#L.subs({A:X, B:Wtemp}).diff(X)[0,0][0,0].replace(n,vN).replace(sigmaApply, sigmaApply_)#.doit()
#diff(L.replace(A,A.T), A) # ERROR max recursion depth exceeded

# %% codecell
#Lmat = L.subs({A:X, B:Wtemp}).diff(X).subs({X:A, Wtemp: B}); Lmat #replace(X, A).replace(Y,B); Lmat
# NOTE need to do replace at the end (instead of subs) else it says unhasable type mutabledensematrix.
Lmat = L.subs({A:X, B:Wtemp}).diff(X).replace(X, A).replace(Wtemp,B); Lmat
#L.diff(A) # HELL ON THE EDITOR NEVER TRY THIS AGAIN
# %% codecell
#L.replace(A,X).replace(B,W)

# %% codecell
# Method 2 approach for comparison:
#L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(X)#.subs(specToElemD)
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
