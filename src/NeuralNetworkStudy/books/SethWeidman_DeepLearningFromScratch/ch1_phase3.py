# %% codecell
from sympy import Matrix, Symbol, derive_by_array, Lambda, Function, MatrixSymbol, Derivative
from sympy import var
from sympy.abc import x, i, j, a, b, c, d




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









# %% codecell

def siga(mat: Matrix) -> Matrix:
     #lst = mat.tolist()
     nr, nc = mat.shape

     applied = [[sigma(mat[i,j]) for j in range(0, nc)] for i in range(0, nr)]

     return Matrix(applied)


# way 2 of declaring S (better way)
sigma = Function('sigma')

sigmaApply = Function("sigma_apply") #lambda matrix:  matrix.applyfunc(sigma)

sigmaApply_ = lambda matrix: matrix.applyfunc(sigma)

sigmaApply_2 = lambda matrix: siga(matrix)

S = sigmaApply(N); S
# %% codecell
sigmaApply_(Nelem)
# %% codecell
sigmaApply_2(Nelem)
# %% codecell
#sigmaApply_2(A*B).diff(Matrix(A))

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

# %% codecell
# %% codecell
# %% codecell
# %% codecell
#L.replace(n,v).diff(A).replace(lambd,lambd_) ### ERROR sigma object is not iterable
#L.replace(n,vL).diff(A).replace(sigmaApply, sigmaApply_)### ERROR
#L.replace(n,v).diff(A).replace(sigmaApply, sigmaApply_) ### ERROR dummy object has no attribute applyfunc
# %% codecell
#L.replace(sigmaApply, sigmaApply_).diff(A) # ERROR
# L.replace(lambd, lambd_) # ERROR

#L.replace(n, v).replace(sigmaApply, sigmaApply_2)# shows matrix results, too specific, want the function composition notation as below but just applied to the function v(X,W) in abstract way
### METHOD 0: (prepare by substituting n --> v, then sigmaApply --> sigma)
L.replace(n, v).replace(sigmaApply, sigmaApply_)#.replace(lambd, lambd_)
# %% codecell

# NOTE: the point here is that even replacing with a sympy Lambda doesn't give same result as above since above uses the V.applyfunc(sigma) within the Lambda.
L.replace(sigmaApply, Lambda(d, sigma(d)))
# %% codecell

vSym = Symbol('v', applyfunc=True)
L.replace(n(A,B), vSym)
# %% codecell
#L.replace(n(A,B), vSym).replace(sigmaApply, sigmaApply_)# ERROR because Symbol has no atttribute applyfunc (that is the one we want though so must use matrix symbol which for some reason works instead of just an ordinary symbol v
#V = MatrixSymbol()
# Takes in the symbols A and B matrices and returns the matrix symbol with the shape that is supposed to result after A*B
V = lambda matA, matB: MatrixSymbol('V', matA.shape[0], matB.shape[1])
V
V(A,B)#.shape
# %% codecell
from sympy import symbols
#V = MatrixSymbol('V', X.shape[0], W.shape[1])
i, j = symbols('i j')
M = MatrixSymbol('M', i, j)# abstract shape
sigmaApply_L = Lambda(M, M.applyfunc(sigma))
lambda_L = Lambda(M, sum(M))
# %% codecell
sigmaApply_L(A)
# %% codecell
# TODO: trying to figure out how to write L so that it is in terms of lambdas so get the form (d ---> sigma(d) COMPOAED ((X,W) -> V)) instead of (sigmaApply(v(X,W)))
Vs = MatrixSymbol("Vs", A.shape[0], B.shape[1])
VL = Lambda((A,B), MatrixSymbol('V', A.shape[0], B.shape[1]))
VL
# %% codecell
L.replace(n, VL)#.replace(sigmaApply, sigmaApply_L).subs({V:VL})

# %% codecell
L.replace(n, VL).replace(sigmaApply, sigmaApply_)#.subs({VL(A,B) : n(A,B)}) ### ERROR
# This is v(X,W) in Lambda form:
VL
# %% codecell
VL(A,B)
#L.subs({n: V})

# %% codecell
L.replace(n(A,B), VL(A,B))#.replace(sigmaApply, sigmaApply_).subs({V(A,B) : n})
# %% codecell
lambd(sigmaApply(VL))
# %% codecell
lambd(sigmaApply(VL)).replace(VL, n(A,B))
# %% codecell
lambd(sigmaApply(VL)).diff(A)
# %% codecell
lambd(sigmaApply(VL)).diff(A).replace(VL, n(A,B))
# %% codecell
lambd(sigmaApply(VL))#.replace(sigmaApply, sigmaApply_)#replace(V, n(A,B)).replace(sigmaApply, sigmaApply_)
# %% codecell
# GOAL: want both sigma_apply to be in ---> form composed with the above x,w ---> V form
#lambd(sigmaApply(V)).replace(V, Vs).replace(sigmaApply, sigmaApply_).replace(Vs, V(A,B))### ERROR
lambd(sigmaApply(n(A,B))).replace(n(A,B), VL)
sigmaApply_(A)
sigmaApply_L(A)
# %% codecell
sigmaApply(Vs).replace(sigmaApply, sigmaApply_)
# %% codecell
sigmaApply(VL(A,B)).replace(sigmaApply, sigmaApply_)#.replace(V(A,B), V)#.subs({sigmaApply: sigmaApply_L})

# %% codecell
#sigmaApply(Vs).subs({Vs : V, sigmaApply: sigmaApply_L}) ### ERROR must be matrix instance
#sigmaApply(Vs).replace(sigmaApply , sigmaApply_L).subs({Vs : V})
#sigmaApply(V).replace(sigmaApply, sigmaApply_L)

# %% codecell

sa = Lambda((A,B), VL)
sa
# %% codecell
### ALTERNATE try of declaring a sigma-apply kind of function
#sas = Lambda((A,B), Vs.applyfunc(sigma))
# %% codecell
Lambda((A,B), sigma(VL))
# %% codecell
Lambda((A,B), sigma(VL)).diff(A) # nothing useful with this format, and weird-wrong since doesn't do chain rule wi.r. to sigma

# %% codecell
Lambda((A,B), sigma(VL(A,B)))
# %% codecell
sas = Lambda((A,B), VL(A,B).applyfunc(sigma))

sas
# %% codecell
# YAY this works now I can replace MATRIX SYMBOLS with ordinary sympy LAMBDAS (replace cano only replace same kind of thing / type)
sigma(Vs).subs(Vs, VL)
#
# %% codecell
sas(A,B)
# %% codecell
# A.applyfunc(sigma).subs(A, VL)# subs method doesn't work here with applyfunc
L
# %% codecell
#sas(A,B).replace(V, V(A,B))

# %% codecell
sigmaApply_L

# %% codecell
sigmaApply_L(M)
# %% codecell
#sigmaApply_LFake = Lambda(M, M.applyfunc(sigma))
sigmaApply(M).replace(sigmaApply, sigmaApply_L)
# %% codecell
#sigmaApply(M).replace(sigmaApply, sigmaApply_).subs(M, n(A,B))
n = Function("v", applyfunc=True)
#sigmaApply_(Vs.subs(Vs, Lambda((A,B), n(A,B))))
from sympy import lambdify
sigma(lambdify([A,B], n(A,B)))

inner = Lambda((A,B), n(A,B)); inner

#sigmaApply_(n(A,B))
sigmaApply(inner).replace(sigmaApply, Lambda(A, sigma(A)))
# %% codecell
#sigmaApply_L(M).subs(M, inner)
Lambda(d, sigma(d))
# %% codecell
### CLOSEST ever gotten to function composition (?) with sympy ....
Lambda(d, sigma(inner))


# %% codecell
Lambda(d, sigma(inner)).diff(A)
# %% codecell
#Lambda(d, sigma(inner)).replace(inner, vL(A,B)).diff(A)

# %% codecell
# %% codecell
# %% codecell
# sigmaApply_L(M).subs(M, VL)# new subs method fails here too
#sigmaApply_(M).subs(M, VL)
# %% codecell
sigmaApply_L(M).diff(M)
# %% codecell
# %% codecell
# %% codecell
sigma(VL)#.replace(V, V(A,B))
# %% codecell
sigma(VL).replace(VL, VL(A,B))
# %% codecell
#sigma(V).replace(V, VL)

# %% codecell
# %% codecell

# %% codecell
f = Function('f')
xtoxL = Lambda(a, a)
xtox = lambda a: a

f(x).subs({x : xtoxL})
# %% codecell
f(x).subs(x, xtox)# works but below one with replace doesn't. When replacing arg with function uses SUBS without dictionary (instead of replace)
# %% codecell
# f(x).replace(x, xtox)### ERROR xtox expects one positional argument ( I think replace only replaces the same kind of thing, never for instance a matrix symbol for a function or vice versa. the replacement needs to be of the same type / kind. But Lambda seems to work (as above))
f(x).replace(x, xtoxL)
# %% codecell
# %% codecell
# %% codecell
#lambd(sigmaApply(n(A,B))).replace(n(A,B), Vs).replace(sigmaApply, sigmaApply_).replace(Vs, V)# ### ERROR rec replace must be matrix instance ....


# %% codecell
# %% codecell

# %% codecell
# %% codecell
### METHOD 0: the matrix diff rule in the most abstract form possible
n = Function("v", applyfunc=True) # necessary
L = lambd(sigmaApply(n(A,B)))

lambd_L = Lambda(A, sum(A))

lambd_L(A)
# %% codecell
lambd_L(sigmaApply(n(A,B)))#.replace(n, vL).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_L)

# %% codecell
L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(A)
# %% codecell
### SUCCESS! We see now that the matrix chain rule indeed makes the X transpose factor out on the left!!! (while compared to the above, the matrix transpose W^T factors out on the right, just like the book says (page 45 in the NOTE section of Seth Weidman book))
L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B)
# %% codecell
# Not showing ???
L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B).replace(lambd, lambd_L)
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
# %% markdown [markdown]
# Compare the above matrix symbol way with the Lsum way:
#
# ### END RESULT of METHOD 1:
# %% codecell
#Lsum = L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)

L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(X)#.subs(specToElemD)
# %% codecell


# %% markdown [markdown]
# COMPARING METHOD 0 (abstract way) with METHOD 2 (direct way) when differentiating .w.r.t to X vs. w.r.t to W
# ### With respect to X (abstract)
# %% codecell
L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(A)

# %% markdown [markdown]
# ### With respect to X (direct)
# %% codecell
L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(X).subs(specToElemD)

# %% markdown [markdown]
# ### With respect to W (abstract)
# %% codecell
L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(B)

# %% markdown [markdown]
# ### With respect to W (direct)
# %% codecell
L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(W).subs(specToElemD)
# %% codecell
# %% codecell
# %% codecell
# %% codecell
# %% markdown [markdown]
# ### NEXT: try to substitute the X, W matrices step by step to see if you can come to the same result as the direct forms above (from method 2 or 1)
# %% codecell
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

# %% codecell
#L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(A).replace(A,Matrix(A))##ERROR noncommutative matrix scalars
# WANT: to be able to do diff and have the expression come out as above with X^T on left and W^T on right, when using just this form, with abstract form v:
L.replace(A,A.T).replace(B,B.T)
# %% codecell
# Error if applying sigma to the v function because it sais v has no attribute applyfunc to trying now to making it have the attriute applyfunc.
y = Function('y', applyfunc=True, real=True)



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

#siga2 = Lambda(a, siga(a))
# %% codecell
Ly.replace(A, A.T).replace(B, b).diff(b)#.replace(sigmaApply, siga)
# %% codecell
L.replace(n, vN).replace(sigmaApply, sigmaApply_).subs(elemToMatArgD)
# %% codecell
#L.replace(n, vN).replace(sigmaApply, sigmaApply_).subs(elemToMatArgD).diff(A)## ERROR: max recursion depth eceeded

L.replace(n, vN).replace(sigmaApply, sigmaApply_).subs(elemToMatArgD).diff(Matrix(3,2,list(elemToMatArgD.values())))
# %% codecell
A.applyfunc(sigma)
# %% codecell
sigma = Function("sigma", applyfunc=True, bool=False)
# %% codecell
sigma.__dict__
# %% codecell
Ly = lambd(sigmaApply(y(A,B))); Ly
# %% codecell
(X*W).applyfunc(sigma)
# %% codecell
(A*B).applyfunc(sigma)
# %% codecell
siga(A)
#A.applyfunc(siga) ### ERROR dumy object has no attribute shape
# %% codecell
y = Function("y", applyfunc = True, bool=False, shape=(3,3))
y.shape
# %% codecell
# siga(y(A,B))### ERROR: function y is not subscriptable
# %% codecell
# %% codecell
# %% codecell
# %% codecell
# %% codecell
Ly.subs({A:a,B:b}).diff(b).subs({a:A, b:B})#.replace(sigmaApply, sigmaApply_)
# %% codecell
L.replace(A,a).replace(B,b).diff(b).subs({a:A,b:B})#.replace(sigmaApply, sigmaApply_)#.diff(b)
# %% codecell
sigma = Function("sigma", applyfunc=True, real=True)
sigmaApply_ = lambda mat: mat.applyfunc(sigma)
L = lambd(sigmaApply(n(A,B)))

#L.replace(A,a).replace(B,b).diff(b).subs({a:A,b:B}).replace(sigmaApply, sigmaApply_)
L.replace(n, v).replace(sigmaApply, sigmaApply_).diff(A)
#m = Symbol("m", shape=(3,2))
#m.shape

#sigmaApply_3 = Lambda(m, siga(m))

#L.replace(A,a).replace(B,b).diff(b).replace(b,B).replace(a,A).subs({n:vL}).replace(sigmaApply, sigmaApply_2) ### ERROR: Dummy object has no attribute shape

# %% codecell
# Ly.replace(B, b).diff(A)#.replace(sigmaApply, siga)### ERROR noncommutative matrix scalars not supported
Ly.replace(A, A.T).replace(B, b).diff(b).replace(b, B).replace(A.T, A)#.replace(sigmaApply, siga)
# %% codecell
#Ly.replace(B,b).diff(b).replace(b,B) ### ERROR
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
#elem2.subs(elemToMatArgD).doit()#ERROR max recursion depth exceeeded
