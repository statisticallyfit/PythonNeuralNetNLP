# Review: Jacobian Matrix

```python title="codecell"
import sys
import os

PATH: str = '/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP'

NEURALNET_PATH: str = PATH + '/src/MatrixCalculusStudy'

sys.path.append(PATH)
sys.path.append(NEURALNET_PATH)
```

```python title="codecell"
from sympy import Matrix, MatrixSymbol, Symbol, derive_by_array, diff, sin, exp, symbols, Function
from sympy.abc import i, j
```


```python
from src.utils.GeneralUtil import *
from src.MatrixCalculusStudy.MatrixDerivLib.symbols import Deriv
from src.MatrixCalculusStudy.MatrixDerivLib.diff import diffMatrix
from src.MatrixCalculusStudy.MatrixDerivLib.printingLatex import myLatexPrinter

from IPython.display import display, Math
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax', latex_printer= lambda e, **kw: myLatexPrinter.doprint(e))
```

### Jacobian Matrix and Multivariable Functions
A vector $\mathbf{f} = \big( f_1, f_2, ..., f_m \big)$ of $m$ functions, each depending on $n$ variables $\mathbf{x} = \big(x_1, x_2, ..., x_n \big)$ defines a transformation or function from $\mathbb{R}^n$ to $\mathbb{R}^m$. Specifically, if $\mathbf{x} \in \mathbb{R}^n$ and if:
$$
y_1 = f_1 \big(x_1,x_2,...,x_n \big) \\
y_2 = f_2 \big(x_1,x_2,...,x_n \big) \\
\vdots \\
y_m = f_m \big(x_1,x_2,...,x_n \big)
$$
then $\mathbf{y} = \big(y_1, y_2, ..., y_m \big)$ is the point in $\mathbb{R}^m$ that corresponds to $\mathbf{x}$ under the transformation $\mathbf{f}$. We can write these equations more compactly as:
$$
\mathbf{y} = \mathbf{f}(\mathbf{x})
$$
Information about the rate of change of $\mathbf{y}$ with respect to $\mathbf{x}$ is contained in the various partial derivatives $\frac{\partial y_i}{\partial x_j}$ for $1 \leq i \leq m, 1 \leq j \leq n$ and is conveniently organized into an $m \times n$ matrix $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ called the **Jacobian matrix** of the transformation $\mathbf{f}$. The Jacobian matrix is the collection of all $m \times n$ possible partial derivatives ($m$ rows and $n$ columns), which is the stack of $m$ gradients with respect to $\mathbf{x}$:
$$
\Large
\begin{aligned}
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} &= \begin{pmatrix}
   \nabla f_1(\mathbf{x}) \\
   \nabla f_2(\mathbf{x}) \\
   \vdots \\
   \nabla f_m(\mathbf{x})
\end{pmatrix}
= \begin{pmatrix}
   \frac{\partial}{\partial \mathbf{x}} f_1(\mathbf{x}) \\
   \frac{\partial}{\partial \mathbf{x}} f_2(\mathbf{x}) \\
   \vdots \\
   \frac{\partial}{\partial \mathbf{x}} f_m(\mathbf{x})
\end{pmatrix} \\
&= \begin{pmatrix}
  \frac{\partial}{\partial x_1} f_1(\mathbf{x}) & \frac{\partial}{\partial x_2} f_1(\mathbf{x}) & ... & \frac{\partial}{\partial x_n} f_1(\mathbf{x}) \\
  \frac{\partial}{\partial x_1} f_2(\mathbf{x}) & \frac{\partial}{\partial x_2} f_2(\mathbf{x}) & ... & \frac{\partial}{\partial x_n} f_2(\mathbf{x}) \\
  \vdots & \vdots &  & \vdots \\
  \frac{\partial}{\partial x_1} f_m(\mathbf{x}) & \frac{\partial}{\partial x_2} f_m(\mathbf{x}) & ... & \frac{\partial}{\partial x_n} f_m(\mathbf{x})
\end{pmatrix} \\

\frac{\partial \mathbf{y}}{\partial \mathbf{x}} &= \begin{pmatrix}
  \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & ... & \frac{\partial f_1}{\partial x_n} \\
  \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & ... & \frac{\partial f_2}{\partial x_n} \\
  \vdots & \vdots &  & \vdots \\
  \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & ... & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}
\end{aligned}
$$
This linear transformation represented by the Jacobian matrix is called **the derivative** of the transformation $\mathbf{f}$.

Each $\frac{\partial f_i}{\partial \mathbf{x}}$ is a horizontal $n$-vector because the partial derivative is with respect to a vector $\mathbf{x}$ whose length is $n = |\mathbf{x}|$, making the width of the Jacobian $n$ (there are $n$ parameters that are variable, each potentially changing the function's value).


```python
X = Matrix(MatrixSymbol('x', 3,3))
X
```
```python
W = Matrix(MatrixSymbol('w', 3,2))
W
```
```python title="codecell"
X*W
```

```python title="codecell"
derive_by_array(X*W, X)
```
```python title="codecell"
(X*W).diff(X)



```

```python title="codecell"

x, y, z = symbols('x y z')
f, g, h = list(map(Function, 'fgh'))

xv = x,y,z
xv
```
```python
yv = [f(*xv), g(*xv), h(*xv)]
yv
```
```python title="codecell"
Matrix(yv)
```

```python title="codecell"
#display(Matrix(yv).jacobian(xv))
Matrix(yv).jacobian(Matrix(xv))
#display(yv.jacobian(xv))
```

```python title="codecell"
derive_by_array(yv, xv)
```

```python title="codecell"
assert Matrix(derive_by_array(yv, xv)).transpose() == Matrix(yv).jacobian(xv)
```

```python title="codecell"
### TEST 2: substituting values
m = Matrix(yv).jacobian(xv)
m.subs({x:1, y:2, z:3})
```

```python title="codecell"
m.subs({f(*xv):x**2 * y*z, g(*xv):sin(x*y*z*3), h(*xv):y + z*exp(x)})
```

```python title="codecell"
m_subs = m.subs({f(*xv):x**2 * y*z, g(*xv):sin(x*y*z*3), h(*xv):y + z*exp(x)})

m_subs.doit()
```

```python title="codecell"
m_subs.doit().subs({x:1, y:2, z:3})


```

```python title="codecell"
# More general / abstract example:

n,m = 5,7

xv = Matrix(n, 1, lambda i,j : var_i('x', i+1))

fs = Matrix(m, 1, lambda i,_ : var_i('f', i+1))

fv = Matrix(m, 1, lambda i,_: func_i('f', i, xLetter = 'x', xLen = n))

mapFFuncToF = dict(zip(fv, fs))
mapFToFFunc = dict(zip(fs, fv))

showGroup([xv, fv, fs])
```

```python title="codecell"
fv.jacobian(xv)

# The final jacobian (simplified)
jacF = fv.jacobian(xv).subs(mapFFuncToF)
jacF
```


```python title="codecell"

# Doing it the derive_by_array way
import itertools

fv_list = list(itertools.chain(*fv.tolist()))
xv_list = list(itertools.chain(*xv.tolist()))


jacF_derive = Matrix(derive_by_array(fv_list, xv_list)).transpose().subs(mapFFuncToF)

jacF_derive
```

```python title="codecell"
assert jacF == jacF_derive
```
```python title="codecell"

```
