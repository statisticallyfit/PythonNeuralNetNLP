```python title="codecell"

from sympy import diff, sin, exp, symbols, Function, Matrix, MatrixSymbol, FunctionMatrix, derive_by_array


from sympy import Symbol


x, y, z = symbols('x y z')
f, g, h = list(map(Function, 'fgh'))
```

```python title="codecell"
# 1) manualy declaration of vector variables
xv = x,y,z
#f(xv).subs({x:1, y:2,z:3})
Matrix(xv)
```
```python title="codecell"
yv = [f(*xv), g(*xv), h(*xv)]; yv
```
```python title="codecell"
Matrix(yv)
```


```python title="codecell"
from sympy.abc import i,j

# 2) Dynamic way of declaring the vector variables
def var(letter: str, i: int) -> Symbol:
    letter_i = Symbol('{}_{}'.format(letter, i+1), is_commutative=True)
    return letter_i

n,m,p = 5,7,4

xv = Matrix(n, 1, lambda i,j : var('x', i)); xv
```

```python title="codecell"
def func(i):
    y_i = Function('y_{}'.format(i+1))(*xv)
    return y_i

yv = Matrix( m, 1, lambda i,_:  func(i)); yv
```


<!-- #region markdown -->
### Gradient Vector
Let $f(\mathbf{x})$ be a differentiable real-valued function of the real $m \times 1$ vector $\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_m \end{pmatrix}$.

Then the **vector of first order partial derivative**s $\frac{\partial f}{\partial \mathbf{x}}$, also called the **gradient vector**, is defined as:
$$
\frac{\partial f}{\partial \mathbf{x}} = \Large\begin{pmatrix}
   \frac{\partial f}{\partial x_1} \\
   \frac{\partial f}{\partial x_2} \\
   \vdots \\
   \frac{\partial f}{\partial x_m}
\end{pmatrix}
$$

The **vector of first order partial derivative**s $\frac{\partial f}{\partial \mathbf{x}^T}$ is defined as:
$$
\frac{\partial f}{\partial \mathbf{x}^T}
= \Bigg( \frac{\partial f}{\partial \mathbf{x}} \Bigg)^T
= \Large\begin{pmatrix}
   \frac{\partial f}{\partial x_1} &
   \frac{\partial f}{\partial x_2} & ... &
   \frac{\partial f}{\partial x_m}
\end{pmatrix}
$$
<!-- #endregion -->
```python title="codecell"
# ### for deriv of scalar-valued multivariate function with respect to the vector

f(*xv).diff(xv)
```

```python title="codecell"
derive_by_array(f(*xv), xv)
```


```python title="codecell"
assert Matrix(derive_by_array(f(*xv), xv)) == f(*xv).diff(xv)


```

<!-- #region markdown -->
### Derivative of Vector with Respect to Scalar
Let $\mathbf{y}(x) = \begin{pmatrix} y_1(x) \\ y_2(x) \\ \vdots \\ y_m(x) \end{pmatrix}$ be a vector of order $m$, where each of the elements $y_i$ are functions of the scalar variable $x$. Specifically, $y_i = f_i(x), 1 \leq i \leq m$, where $f_i : \mathbb{R} \rightarrow \mathbb{R}$ and $\mathbf{y} : \mathbb{R} \rightarrow \mathbb{R}^m$.

Then the **derivative of the vector $\mathbf{y}$ with respect to scalar $x$** is defined as:
$$
\frac{\partial \mathbf{y}}{\partial x} = \Large\begin{pmatrix}
   \frac{\partial y_1}{\partial x} & \frac{\partial y_2}{\partial x} & ... & \frac{\partial y_m}{\partial x}
\end{pmatrix}
$$
<!-- #endregion -->

```python title="codecell"
# ### for deriv of a vector-valued function by its scalar argument
#yv = [f(x), g(x), h(x)]; yv
from sympy.abc import x

yv = Matrix( 1, m, lambda _, j:  Function('y_{}'.format(j+1))(x)); yv


```

```python title="codecell"
yv.diff(x)

 # NOTE: incorrect shape (is column-wise, must be row-wise like below) when defining the yv matrix to be m x 1 instead of 1 x m. Ideally want to define a regular m x 1 y-vector of functions y_i and to have the diff by x to be 1 x m.
```

```python title="codecell"
derive_by_array(yv, x) # Correct shape (row-wise)
```
```python title="codecell"
Matrix(derive_by_array(yv, x))
```
```python title="codecell"
assert Matrix(derive_by_array(yv, x)) == Matrix(yv).diff(x)


```

<!-- #region markdown -->
### Vector Chain Rule
<!-- #endregion -->
```python title="codecell"
# ### for vector chain rule
```
