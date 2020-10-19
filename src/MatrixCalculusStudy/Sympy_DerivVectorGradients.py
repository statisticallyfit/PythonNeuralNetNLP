# %% markdown
# # Review: Gradient and Vector Chain Rule

# %% codecell
from sympy import diff, sin, exp, symbols, Function, Matrix, MatrixSymbol, FunctionMatrix, derive_by_array, Symbol



# %%
import os, sys 


PATH: str = '/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP'

UTIL_DISPLAY_PATH: str = PATH + "/src/utils/GeneralUtil/"

NEURALNET_PATH: str = PATH + '/src/NeuralNetworkStudy/books/SethWeidman_DeepLearningFromScratch'

sys.path.append(UTIL_DISPLAY_PATH)
sys.path.append(PATH)
sys.path.append(NEURALNET_PATH)

# %%
from src.utils.GeneralUtil import *
from src.MatrixCalculusStudy.MatrixDerivLib.symbols import Deriv
from src.MatrixCalculusStudy.MatrixDerivLib.diff import diffMatrix
from src.MatrixCalculusStudy.MatrixDerivLib.printingLatex import myLatexPrinter

from IPython.display import display, Math
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax', latex_printer= lambda e, **kw: myLatexPrinter.doprint(e))

# %%
x, y, z = symbols('x y z')
f, g, h = list(map(Function, 'fgh'))

# %% codecell
# 1) manualy declaration of vector variables
xv = x,y,z
#f(xv).subs({x:1, y:2,z:3})
#Matrix(xv)
Matrix(xv)
# %% codecell
yv = [f(*xv), g(*xv), h(*xv)]

Matrix(yv)




# %% codecell
n,m,p = 5,7,4

xv = Matrix(n, 1, lambda i,j : var_i('x', i+1))
xv

# %% codecell
yv = Matrix( m, 1, lambda i,_:  func_i('y', i, xLetter = 'x', xLen = n))
yv


# %% markdown
# ### Derivative of Real-Valued, Multivariate Function with Respect to Vector: The Gradient Vector
#
# Let the multivariate and real-valued function $f(\mathbf{x}) = f(x_1,x_2,...,x_n)$ from $\mathbb{R}^n \longrightarrow \mathbb{R}$  be a function of the real $n \times 1$ vector $\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}$.
#
# Then the **vector of first order partial derivative**s $\frac{\partial f}{\partial \mathbf{x}}$, also called the **gradient vector**, is defined as:
#
# * Notation with arguments:
#
# $$
# \nabla f(\mathbf{x}) = \frac{\partial}{\partial \mathbf{x}} f(\mathbf{x})
# {\Large
# = \begin{pmatrix}
#         \frac{\partial }{\partial x_1} f(\mathbf{x}) \\
#         \frac{\partial }{\partial x_2} f(\mathbf{x}) \\
#         \vdots \\
#         \frac{\partial }{\partial x_n} f(\mathbf{x})
# \end{pmatrix} }
# $$
#
# * Notation without arguments:
#
# $$
# \nabla f = \frac{\partial f}{\partial \mathbf{x}} = \Large\begin{pmatrix}
#    \frac{\partial f}{\partial x_1} \\
#    \frac{\partial f}{\partial x_2} \\
#    \vdots \\
#    \frac{\partial f}{\partial x_n}
# \end{pmatrix}
# $$




# %% markdown
# ### Vector of First Order Partial Derivatives
#
# The **vector of first order partial derivative**s $\frac{\partial f}{\partial \mathbf{x}^T}$ is defined as the transpose of the **gradient** of $f$:
# $$
# \frac{\partial f}{\partial \mathbf{x}^T}
# = (\nabla f)^T
# = \Bigg( \frac{\partial f}{\partial \mathbf{x}} \Bigg)^T
# = \Large\begin{pmatrix}
#    \frac{\partial f}{\partial x_1} &
#    \frac{\partial f}{\partial x_2} & ... &
#    \frac{\partial f}{\partial x_m}
# \end{pmatrix}
# $$






# %% codecell
# ### for deriv of scalar-valued multivariate function with respect to the vector

#f(*xv).diff(xv)

f(*xv).diff(xv)

# %% codecell
#derive_by_array(f(*xv), xv)
derive_by_array(f(*xv), xv)

# %% codecell
assert Matrix(derive_by_array(f(*xv), xv)) == f(*xv).diff(xv)



# %% markdown
# ### Derivative of Vector-Valued, Single Variable Function with Respect to Scalar
# Let $\mathbf{y}(x) = \begin{pmatrix} y_1(x) \\ y_2(x) \\ \vdots \\ y_m(x) \end{pmatrix}$ be a vector of order $m$, where each of the elements $y_i$ are functions of the scalar variable $x$. Specifically, $y_i = f_i(x), 1 \leq i \leq m$, where $f_i : \mathbb{R} \rightarrow \mathbb{R}$ and $\mathbf{y} : \mathbb{R} \rightarrow \mathbb{R}^m$.
#
# Then the **derivative of the vector-valued function $\mathbf{y}$ with respect to its ginle variable $x$** is defined as:
# $$
# \frac{\partial \mathbf{y}}{\partial x} = \Large\begin{pmatrix}
#    \frac{\partial y_1}{\partial x} & \frac{\partial y_2}{\partial x} & ... & \frac{\partial y_m}{\partial x}
# \end{pmatrix}
# $$

# %% codecell
# ### for deriv of a vector-valued function by its scalar argument
#yv = [f(x), g(x), h(x)]; yv
from sympy.abc import x

yv = Matrix( 1, m, lambda _, j:  Function('y_{}'.format(j+1))(x))
yv


# %% codecell
yv.diff(x)

 # NOTE: incorrect shape (is column-wise, must be row-wise like below) when defining the yv matrix to be m x 1 instead of 1 x m. Ideally want to define a regular m x 1 y-vector of functions y_i and to have the diff by x to be 1 x m.

# %% codecell
#derive_by_array(yv, x) # Correct shape (row-wise)
#display(derive_by_array(yv, x))
# NOTE: this displays double matrix dimension, so no need for it here, need to convert result to matrix as below
# %% codecell
#Matrix(derive_by_array(yv, x))
Matrix(derive_by_array(yv, x))
# %% codecell
assert Matrix(derive_by_array(yv, x)) == Matrix(yv).diff(x)





# %% markdown
# ### Vector Chain Rule
# In general the Jacobian matrix of the composition of two vector-valued functions of a vector variable is the matrix product of their Jacobian matrices.
#
# To see this let $\mathbf{z} = \mathbf{f}(\mathbf{y})$ be a transformation from $\mathbb{R}^k$ to $\mathbb{R}^m$ given by:
# $$
# z_1 = f_1 \big(y_1,y_2,...,y_k \big) \\
# z_2 = f_2 \big(y_1,y_2,...,y_k \big) \\
# \vdots \\
# z_m = f_m \big(y_1,y_2,...,y_k \big)
# $$
# which has the $m \times k$ Jacobian matrix:
# $$
# \Large
# \frac{\partial \mathbf{f}}{\partial \mathbf{y}} = \begin{pmatrix}
#   \frac{\partial f_1}{\partial y_1} & \frac{\partial f_1}{\partial y_2} & ... & \frac{\partial f_1}{\partial y_k} \\
#   \frac{\partial f_2}{\partial y_1} & \frac{\partial f_2}{\partial y_2} & ... & \frac{\partial f_2}{\partial y_k} \\
#   \vdots & \vdots &  & \vdots \\
#   \frac{\partial f_m}{\partial y_1} & \frac{\partial f_m}{\partial y_2} & ... & \frac{\partial f_m}{\partial y_k}
# \end{pmatrix}
# $$
#

# and let $\mathbf{y} = \mathbf{g}(\mathbf{x})$ be another such transformation from $\mathbb{R}^n$ to $\mathbb{R}^k$ given by:
#
# $$
# y_1 = g_1 \big(x_1,x_2,...,x_n \big) \\
# y_2 = g_2 \big(x_1,x_2,...,x_n \big) \\
# \vdots \\
# y_k = g_k \big(x_1,x_2,...,x_n \big)
# $$
# which has the $k \times n$ Jacobian matrix:
# $$
# \Large
# \frac{\partial \mathbf{g}}{\partial \mathbf{x}} = \begin{pmatrix}
#   \frac{\partial g_1}{\partial x_1} & \frac{\partial g_1}{\partial x_2} & ... & \frac{\partial g_1}{\partial x_n} \\
#   \frac{\partial g_2}{\partial x_1} & \frac{\partial g_2}{\partial x_2} & ... & \frac{\partial g_2}{\partial x_n} \\
#   \vdots & \vdots &  & \vdots \\
#   \frac{\partial g_k}{\partial x_1} & \frac{\partial g_k}{\partial x_2} & ... & \frac{\partial g_k}{\partial x_n}
# \end{pmatrix}
# $$
#
# Then the composition $\mathbf{z} = (\mathbf{f} \circ \mathbf{g})(\mathbf{x}) = \mathbf{f}(\mathbf{g}(\mathbf{x}))$ given by :
# $$
# z_1 = f_1 \big( g_1 \big( x_1,...,x_n \big),..., g_k \big( x_1,...,x_n \big) \big) \\
# z_2 = f_2 \big( g_1 \big( x_1,...,x_n \big),..., g_k \big( x_1,...,x_n \big) \big) \\
# \vdots \\
# z_k = f_m \big( g_1 \big( x_1,...,x_n \big),..., g_k \big( x_1,...,x_n \big) \big)
# $$
#
# has, according to the Chain Rule, the $m \times n$ Jacobian matrix
#
# $$
# \Large
# \begin{aligned}
#
# \frac{\partial}{\partial \mathbf{x}} \mathbf{f} \big( \mathbf{g}(\mathbf{x}) \big) &= \frac{\partial \mathbf{f}}{\partial \mathbf{g}} \times \frac{\partial \mathbf{g}}{\partial \mathbf{x}} \\
#
# \begin{pmatrix}
# \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & ... & \frac{\partial f_1}{\partial x_n} \\
# \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & ... & \frac{\partial f_2}{\partial x_n} \\
# \vdots & \vdots & & \vdots \\
# \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & ... & \frac{\partial f_m}{\partial x_n}
# \end{pmatrix}
#
# &= \begin{pmatrix}
# \frac{\partial f_1}{\partial g_1} & \frac{\partial f_1}{\partial g_2} & ... & \frac{\partial f_1}{\partial g_k} \\
# \frac{\partial f_2}{\partial g_1} & \frac{\partial f_2}{\partial g_2} & ... & \frac{\partial f_2}{\partial g_k} \\
# \vdots & \vdots & & \vdots \\
# \frac{\partial f_m}{\partial g_1} & \frac{\partial f_m}{\partial g_2} & ... & \frac{\partial f_m}{\partial g_k}
# \end{pmatrix}
#
# \times
#
# \begin{pmatrix}
# \frac{\partial g_1}{\partial x_1} & \frac{\partial g_1}{\partial x_2} & ... & \frac{\partial g_1}{\partial x_n} \\
# \frac{\partial g_2}{\partial x_1} & \frac{\partial g_2}{\partial x_2} & ... & \frac{\partial g_2}{\partial x_n} \\
# \vdots & \vdots & & \vdots \\
# \frac{\partial g_k}{\partial x_1} & \frac{\partial g_k}{\partial x_2} & ... & \frac{\partial g_k}{\partial x_n}
# \end{pmatrix}
#
# \end{aligned}
# $$
# where $\times$ denotes matrix multiplication, and $m = |\mathbf{f}|, n = |\mathbf{x}|$ and $k = |\mathbf{g}|$.
#
#
# **SOURCES:**
# * R.A Adams - Calculus: A Complete Course (sections 12.5 and 12.6)
# * Thomas Weir - Calculus (section 14.4)
# * [Medium's blog post on "The Matrix Calculus you Need for Deep Learning"](https://medium.com/@rohitrpatil/the-matrix-calculus-you-need-for-deep-learning-notes-from-a-paper-by-terence-parr-and-jeremy-4f4263b7bb8)


# %% codecell
n, k, m = 3, 4, 5

xv = Matrix(n, 1, lambda i,j : var_i('x', i+1))
xv

# %% codecell
gv = Matrix( k, 1, lambda i,_:  func_i('g', i, xLetter = 'x', xLen = n))
gv

# %% codecell

fv = Matrix(m, 1, lambda i,_: func_i('f', i, xLetter = 'y', xLen = k))
fv


# %% codecell
ys = Matrix(k, 1, lambda i,_: var_i('y', i+1))
fs = Matrix(m, 1, lambda i, _: var_i('f', i+1))
gs = Matrix(k, 1, lambda i, _: var_i('g', i+1))

mapYToGFunc = dict(zip(ys, gv))
mapGFuncToY = dict(zip(gv, ys))

zv = fv.subs(mapYToGFunc)

mapFToFGFunc = dict(zip(fs, zv))
mapFGFuncToF = dict(zip(zv, fs))

mapGToGFunc = dict(zip(gs, gv))
mapGFuncToG = dict(zip(gv, gs))

showGroup([
    mapYToGFunc, 
    mapFToFGFunc, 
    mapGToGFunc
])

assert zv.subs(mapFGFuncToF) == fs
assert fv.subs(mapYToGFunc) == zv

assert zv.shape == fv.shape == (m, 1)
assert ys.shape == gv.shape == (k, 1)
assert xv.shape == (n, 1)


# %% codecell
jacG = gv.jacobian(xv)

showGroup([
    jacG, 
    jacG.subs(mapGFuncToG),
    jacG.subs(mapGFuncToY)
])


# %% codecell
jacFY = fv.jacobian(ys)
jacFY
# %%
jacFG = fv.jacobian(ys).subs(mapYToGFunc)
jacFG

# %% codecell
jacComposed = fv.subs(mapYToGFunc).jacobian(xv)
# %%
jacComposed

# %% codecell
df_dg = jacFG.subs(mapFGFuncToF).subs(mapGFuncToG)
dg_dx = gv.jacobian(xv).subs(mapGFuncToG)
dfg_x = jacComposed.subs(mapFGFuncToF).subs(mapGFuncToG)

showGroup([
    df_dg, 
    dg_dx, 
    dfg_x
])

# %% codecell
# The final test:
assert dfg_x == df_dg * dg_dx
# %% codecell
