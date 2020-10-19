# %% markdown
# # Derivations For Matrix Derivative Rule of $L = \lambda( \sigma_{\text{apply}}( \nu(X, W) ) )$

# %%
from sympy import Matrix, Symbol, derive_by_array, Lambda, Function, MatrixSymbol, Identity, Derivative, symbols, diff
from sympy.abc import x, i, j, a, b

# %%
from typing import *
import sys
import os

PATH: str = '/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP'

UTIL_DISPLAY_PATH: str = PATH + "/src/utils/GeneralUtil/"

NEURALNET_PATH: str = PATH + '/src/NeuralNetworkStudy/books/SethWeidman_DeepLearningFromScratch'

#os.chdir(PATH)
#assert os.getcwd() == NEURALNET_PATH

sys.path.append(PATH)
#assert PATH in sys.path

sys.path.append(UTIL_DISPLAY_PATH)
#assert UTIL_DISPLAY_PATH in sys.path

sys.path.append(NEURALNET_PATH)#
#assert NEURALNET_PATH in sys.path



# %% codecell
from src.utils.GeneralUtil import *
from src.MatrixCalculusStudy.MatrixDerivLib.symbols import Deriv
from src.MatrixCalculusStudy.MatrixDerivLib.diff import diffMatrix
from src.MatrixCalculusStudy.MatrixDerivLib.printingLatex import myLatexPrinter

from IPython.display import display, Math
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax', latex_printer= lambda e, **kw: myLatexPrinter.doprint(e))

# %%
import itertools

from functools import reduce

from typing import *
# %% codecell
def composeTwoFunctions(f, g):
    return lambda *a, **kw: f(g(*a, **kw))

def compose(*fs):
    return reduce(composeTwoFunctions, fs)



# %% codecell
n,m,p = 3,3,2


xi  = Symbol('xi')
xi_1  = Symbol('xi_1')
beta = Symbol('beta')


X = Matrix(n, m, lambda i,j : var_ij('x', i, j))
W = Matrix(m, p, lambda i,j : var_ij('w', i, j))

A = MatrixSymbol('X',n,m)
B = MatrixSymbol('W',m,p)

# matrix variable for sympy Lambda function arguments
M = MatrixSymbol('M', i, j)# abstract shape
# %% codecell
#compose(sigmaApply)(N).replace(sigmaApply, sigmaApply_).diff(N).subs({N : vN(A,B)}).doit()
# %% codecell
###N = MatrixSymbol("N", n, p)# shape of A*B### use Nelem below


showGroup([
    X, W, Matrix(A)
])

# %% codecell
v = Function("nu",applyfunc=True)
v_ = lambda a,b: a*b
vL = Lambda((a,b), a*b)
vN = lambda mat1, mat2: Matrix(mat1.shape[0], mat2.shape[1], lambda i, j: Symbol("n_{}{}".format(i+1, j+1))); vN

Nelem = vN(X, W)
Nspec = v_(X,W)
N = v(A,B)


showGroup([
    Nelem, Nspec, N
])


# %%
sigma = Function('sigma')
sigmaApply = Function("sigma_apply") #lambda matrix:  matrix.applyfunc(sigma)
sigmaApply_ = lambda matrix: matrix.applyfunc(sigma)
sigmaApply_L = Lambda(M, M.applyfunc(sigma))


S = sigmaApply(N)
Sspec = S.subs({A:X, B:W}).replace(v, v_).replace(sigmaApply, sigmaApply_)
Selem = S.replace(v, vN).replace(sigmaApply, sigmaApply_)


showGroup([
    S, Sspec, Selem
])


# %%
lambd = Function("lambda")
lambd_ = lambda matrix : sum(matrix)
#lambda_L = Lambda(M, sum(M))

ABres = MatrixSymbol("R", A.shape[0], B.shape[1])
lambd_L = Lambda(ABres, sum(ABres))

#L = lambd(sigmaApply(v(A,B)))
L = compose(lambd, sigmaApply, v)(A, B)
L



# %% codecell
elemToSpecD = dict(itertools.chain(*[[(Nelem[i, j], Nspec[i, j]) for j in range(p)] for i in range(n)]))
elemToSpec = list(elemToSpecD.items())

specToElemD = {val:key for key, val in elemToSpecD.items()}
specToElem = list(specToElemD.items())

Matrix(elemToSpec)

# %% codecell

elemToSpecFuncD = dict(itertools.chain(*[[(Nelem[i, j], Function("n_{}{}".format(i + 1, j + 1))(Nspec[i, j])) for j in range(p)] for i in range(n)]))

elemToSpecFunc = list(elemToSpecFuncD.items())

specFuncToElemD = {val : key for key , val in elemToSpecFuncD.items()}
specFuncToElem = list(specFuncToElemD.items())

Matrix(elemToSpecFunc)



# %% codecell
elemToNFuncD = dict(itertools.chain(*[[(Nelem[i, j], Function("n_{}{}".format(i + 1, j + 1))(*X,*W)) for j in range(p)] for i in range(n)]))

elemToNFunc = list(elemToNFuncD.items())

nfuncToElemD = {val: key for key, val in elemToNFuncD.items()}
nfuncToElem = list(nfuncToElemD.items())

Matrix(elemToNFunc)



# %% codecell
elemToNmatfuncD = dict(itertools.chain(*[[(Nelem[i, j], Function("n_{}{}".format(i+1,j+1))(A,B) ) for j in range(p)] for i in range(n)]))

elemToNmatfunc = list(elemToNmatfuncD.items())

nmatfuncToElemD = {val: key for key, val in elemToNmatfuncD.items()}
nmatfuncToElem = list(nmatfuncToElemD.items())

Matrix(elemToNmatfunc)



# %% codecell
nmatfuncToSpecD = dict(zip(elemToNmatfuncD.values(), elemToSpecD.values()))

nmatfuncToSpec = list(nmatfuncToSpecD.items())

Matrix(nmatfuncToSpec)





# %% codecell
#L.replace(v,v_).replace(sigmaApply, sigmaApply_).diff(B)

dL_dW_abstract = L.replace(v,v_).replace(sigmaApply, sigmaApply_).diff(B)

showGroup([
    dL_dW_abstract,
    dL_dW_abstract.subs({lambd : lambd_L})
])

# %%
dL_dX_abstract = L.replace(v, v_).replace(sigmaApply, sigmaApply_).diff(A)

dL_dX_abstract


# %% codecell
dL_dW_direct = L.replace(v, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(W).subs(specToElemD)

dL_dW_direct
# %%
dL_dX_direct = L.replace(v, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(X).subs(specToElemD)

dL_dX_direct



# %%
unapplied = sigmaApply_L(vN(A,B))
# Also works: same as above:
#compose(sigmaApply, v)(A,B).replace(v, vN).replace(sigmaApply , sigmaApply_L)
applied = unapplied.doit()

showGroup([
    unapplied,
    applied
])

# %%
dL_dW_step = compose(lambd, sigmaApply, v)(A,B).replace(v, v_).replace(sigmaApply, sigmaApply_).diff(B).subs({A*B : vN(A,B)}).doit()

showGroup([
    dL_dW_step,
    dL_dW_step.replace(unapplied, applied),
    # Carrying out the multplication:
    dL_dW_step.subs({A:X}).doit(), # replace won't work here
    dL_dW_step.subs({A:X}).doit().replace(unapplied, applied)
])



# %%
dL_dX_step = compose(lambd, sigmaApply, v)(A,B).replace(v, v_).replace(sigmaApply, sigmaApply_).diff(A).subs({A*B : vN(A,B)}).doit()


showGroup([
    dL_dX_step,
    dL_dX_step.replace(unapplied, applied),
    dL_dX_step.subs({B:W}).doit(),
    dL_dX_step.subs({B:W}).doit().replace(unapplied, applied)
])



# %% markdown
# Trying to replace further to get the ones matrix for the deriv of lambda expression, but doesn't work, see code below for why (hadamard is not present, just matrix multiplication. Chain rule in this form doesn't know there should be hadamard product between deriv of lambda expression and dsdx expression)
# %%
dle = lambd(xi).diff(xi)
dle_repl = lambd(xi).diff(xi).subs(xi, applied).replace(lambd, lambd_L)

showGroup([
    dle,
    dle_repl
])
# %%
showGroup([
    dL_dW_abstract.replace(sigmaApply_L(A*B), xi),
    dL_dW_abstract.replace(sigmaApply_L(A*B), xi).doit(),
    dL_dW_abstract.replace(sigmaApply_L(A*B), xi).doit().replace(dle, dle_repl) #.doit())
])

# NOTE here it says the matrices are not aligned if we execute doit() to reveal the ones matrix that is dL_dS. True since assumption here is matrix multplication with dL_dS and right hand side, but in fact it is hadamard multiplication.



# %% markdown
# The first part: $\frac{dL}{dS}$
#
# Direct substitution way:
# %%
showGroup([
    lambd(xi).diff(xi).subs(xi, applied),
    lambd(xi).diff(xi).subs(xi, applied).replace(lambd, lambd_L),
    lambd(xi).diff(xi).subs(xi, applied).replace(lambd, lambd_L).doit()
])

# %% markdown
# The substitute into derivative way:
# %%
showGroup([
    lambd(xi).diff(xi).subs(xi, unapplied),
    lambd(xi).diff(xi).subs(xi, unapplied).replace(unapplied, applied),
    # gives same expression as in dldx
    lambd(xi).diff(xi).subs(xi, unapplied).replace(unapplied, applied).replace(lambd, lambd_L)
])



# %% markdown
# The second part: dndx * dsdn
# %% codecell

# %% codecell
dN_dW_times_dS_dN = compose(sigmaApply, v)(A,B).replace(v, v_).replace(sigmaApply, sigmaApply_).diff(B).subs({A*B : vN(A,B)}).doit()

showGroup([
    dN_dW_times_dS_dN,
    dN_dW_times_dS_dN.subs({A:X}), # replace won't work here
    # Carrying out the multplication:
    dN_dW_times_dS_dN.subs({A:X}).doit() # replace won't work here
])

# %%
dN_dX_times_dS_dN = compose(sigmaApply, v)(A,B).replace(v, v_).replace(sigmaApply, sigmaApply_).diff(A).subs({A*B : vN(A,B)}).doit()

showGroup([
    dN_dX_times_dS_dN,
    dN_dX_times_dS_dN.subs({B:W}), # replace won't work here
    # Carrying out the multplication:
    dN_dX_times_dS_dN.subs({B:W}).doit() # replace won't work here
])




# %% codecell
# THis seems right:
dL_dS = lambd(Selem).replace(lambd, lambd_L).diff(Selem)
# ANOTHER WAY: lambd(xi).diff(xi).subs(xi, applied).replace(lambd, lambd_L).doit()

# THIS SEEMS WRONG : ??? how to tell for sure?
#lambd(Selem).diff(Selem).replace(lambd, lambd_L).doit()

dL_dS


# %% codecell

dS_dN = compose(sigmaApply)(M).replace(sigmaApply, sigmaApply_).diff(M).subs({M : vN(A,B)}).doit()

dS_dN_abstract = compose(sigmaApply)(M).replace(sigmaApply, sigmaApply_).diff(M).subs(M, v_(A,B))
# ANOTHER WAY: sigmaApply_L(M).diff(M).subs({M : Nelem}).doit()
# WRONG:
#dS_dN = sigmaApply(Nelem).replace(sigmaApply, sigmaApply_).diff(Matrix(Nelem))
showGroup([
    dS_dN,
    dS_dN_abstract
])


# %% codecell
from sympy import HadamardProduct

dN_dX = B.transpose()
dN_dW = A.transpose()

dS_dW = dN_dW * dS_dN
dS_dW_abstract = compose(sigmaApply, v)(A,B).replace(v, v_).replace(sigmaApply, sigmaApply_).diff(B)

dL_dW = HadamardProduct(dL_dS, dS_dW)

display(dS_dW)
display(dS_dW.subs(A, X).doit())

display(dL_dW)


assert dL_dW == HadamardProduct(dL_dS, dN_dW * dS_dN )


# %%
dL_dW_hadamard = dL_dW.subs(A,X).doit()
# %%
display(dL_dW_abstract)
display(dL_dW_step)
display(dL_dW)
display(dL_dW_hadamard)

# %% markdown
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
LcL = compose(lambd_L, sigmaApply, v)(A, B)
LcL
# %% codecell
symbL = LcL.replace(v, v_).replace(sigmaApply, sigmaApply_)#.diff(A)
symbL
# %% codecell
compose(lambd, sigmaApply, v)(A,B).replace(v,v_).subs({lambd:lambd_L})#.subs({sigmaApply : sigmaApply_L})
# %% codecell
compose(lambd, sigmaApply, v)(A,B).replace(v,v_).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_L)
# %% codecell
compose(lambd, sigmaApply, v)(A,B).replace(v,v_).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_L).doit()
# %% codecell
compose(lambd, sigmaApply, v)(A,B).replace(v,v_).diff(B).doit()#replace(sigmaApply, sigmaApply_)#.replace(lambd, lambd_L).diff(B)
# %% codecell
compose(lambd, sigmaApply, v)(A,B).replace(v,v_).diff(B).replace(lambd, lambd_L)
# %% codecell
compose(lambd, sigmaApply, v)(A,B).replace(lambd, lambd_L)
# %% codecell
compose(lambd, sigmaApply, v)(A,B).replace(lambd, lambd_L).replace(v, v_).replace(sigmaApply, sigmaApply_)
# %% codecell
compose(lambd, sigmaApply, v)(A,B).replace(lambd, lambd_L).replace(v, v_).replace(sigmaApply, sigmaApply_).doit()#.diff(B)
# %% codecell
compose(lambd, sigmaApply, v)(A,B).replace(lambd, lambd_L).replace(v, v_).replace(sigmaApply, sigmaApply_).doit().diff(Matrix(B)).doit()

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
s(v(A,B))
# %% codecell
#sL = lambda x : Lambda(x, sigma(x))
#sL = Lambda(x, lambda x: sigma(x))

sL = Lambda(x, Lambda(x, sigma(x)))
sL
#sL(A)
