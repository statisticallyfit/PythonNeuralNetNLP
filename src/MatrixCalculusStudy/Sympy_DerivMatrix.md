```python

```

```python title="codecell"
from sympy import diff, sin, exp, symbols, Function, Matrix, MatrixSymbol, FunctionMatrix, derive_by_array


from sympy import Symbol


def var(letter: str, i: int, j: int) -> Symbol:
     letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
     return letter_ij

def func(i, j):
     y_ij = Function('y_{}{}'.format(i+1,j+1))(*X)
     return y_ij


n,m,p = 3,3,2

X = Matrix(n, m, lambda i, j: var('x', i, j)); X
```
```python title="codecell"
#Y = MatrixSymbol(Function('y'), 2, 3); Matrix(Y)
#M = MatrixSymbol('M',2,2); Matrix(M)
#Y = Matrix(m, p, lambda i,j: Function('y_{}{}'.format(i+1,j+1))(X) ); Y

Y = Matrix(m, p, lambda i,j:  func(i, j)); Y


```

<!-- #region markdown -->
### Derivative of Matrix With Respect a Matrix
Let $X = \{ x_{ij} \}$ be a matrix of order $m \times n$ and let
$$
y = f(X)
$$
be a scalar function of $X$, so $y \in \mathbb{R}$ and $f: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}$,

Also let the matrix $Y = \{y_{ij}(X) \}$ be of size $p \times q$.

Then we can define the **derivative of $Y$ with respect to $X$** as the following matrix of order $mp \times nq$:

$$
\Large
\frac{\partial Y}{\partial X}
= \begin{pmatrix}
   \frac{\partial Y}{\partial x_{11}} & \frac{\partial Y}{\partial x_{12}} & ... & \frac{\partial Y}{\partial x_{1n}} \\
   \frac{\partial Y}{\partial x_{21}} & \frac{\partial Y}{\partial x_{22}} & ... & \frac{\partial Y}{\partial x_{23}} \\
   \vdots & \vdots & & \vdots \\
   \frac{\partial Y}{\partial x_{m1}} & \frac{\partial Y}{\partial x_{m2}} & ... & \frac{\partial Y}{\partial x_{mn}} \\
\end{pmatrix}
= \Bigg\{ \frac{\partial y_{ij}}{\partial x_{lk}} \Bigg\}
$$
<!-- #endregion -->

```python title="codecell"
# GOT IT this is the definition of gradient matrix (matrix of partial derivatives or dY/dX)
D = derive_by_array(Y, X); D
```
```python title="codecell"
D.subs({Y[0,0]: X[0,0]**2 + X[1,0]}).doit()
```
```python title="codecell"
Y.diff(X) ## GOT IT


```

```python title="codecell"
Yval = Y.subs({Y[0,0]: X[0,0]**2 + X[0,1]*X[1,0] - X[1,1],
        Y[0,1]: X[1,1]**3 + 4* X[0,1] + X[0,0] - X[1,0],
        Y[1,0]: X[1,0] * X[0,0] + 3*X[0,1] * X[1,1],
        Y[1,1]: X[1,1] + X[1,0] + X[0,1] + X[0,0],
        Y[2,0]: 2*X[0,0]**2 * X[0,1] * 3*X[1,0] + 4*X[1,1],
        Y[2,1]: 3*X[0,1] - 5*X[1,1] * X[0,0] - X[1,0]**2})

Yval
```
```python title="codecell"
DYval = D.subs({Y[0,0]: X[0,0]**2 + X[0,1]*X[1,0] - X[1,1],
        Y[0,1]: X[1,1]**3 + 4* X[0,1] + X[0,0] - X[1,0],
        Y[1,0]: X[1,0] * X[0,0] + 3*X[0,1] * X[1,1],
        Y[1,1]: X[1,1] + X[1,0] + X[0,1] + X[0,0],
        Y[2,0]: 2*X[0,0]**2 * X[0,1] * 3*X[1,0] + 4*X[1,1],
        Y[2,1]: 3*X[0,1] - 5*X[1,1] * X[0,0] - X[1,0]**2})
DYval
```
```python title="codecell"
DYval.doit()



```

```python title="codecell"
# ### GOAL: testing the A kronecker B rule for diff of Y = AXB
from sympy import Lambda
l, m, n, q = 3, 5, 4, 2

A = Matrix(l, m, lambda i, j: var('a', i, j))
X = Matrix(m, n, lambda i, j: var('x', i, j))
W = Matrix(n, q, lambda i, j: var('w', i, j))
Y = X*W; Y
```
```python title="codecell"
from sympy.matrices import zeros
E_12 = zeros(m, n)
E_12[1-1,2-1] = 1
E_12
```
```python title="codecell"
Y = X*W; Y
```
```python title="codecell"
E_12*W
```
```python title="codecell"
derive_by_array(Y, X[0,1])
```
```python title="codecell"
assert Matrix(derive_by_array(Y, X[0,1])) == E_12 * W

assert Matrix(derive_by_array(Y, X[0,1])) == Y.diff(X[0,1])
```
