
```python
from sympy import Matrix, Symbol, derive_by_array, Lambda, symbols, Derivative, diff
from sympy.abc import x, y, i, j, a, b
```



# Defining the X
Defining variable-element matrices $X \in \mathbb{R}^{n \times m}$ and $W \in \mathbb{R}^{m \times p}$:


```python
def var(letter: str, i: int, j: int) -> Symbol:
    letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
    return letter_ij


n,m,p = 3,3,2

X = Matrix(n, m, lambda i,j : var('x', i, j)); X
```

```
Matrix([
[x_11, x_12, x_13],
[x_21, x_22, x_23],
[x_31, x_32, x_33]])
```



```python
W = Matrix(m, p, lambda i,j : var('w', i, j)); W
```

```
Matrix([
[w_11, w_12],
[w_21, w_22],
[w_31, w_32]])
```




Defining $N = \nu(X, W) = X \times W$
* $\nu : \mathbb{R}^{(n \times m) \times (m \times p)} \rightarrow \mathbb{R}^{n \times p}$
* $N \in \mathbb{R}^{n \times p}$



```python
v = Lambda((a,b), a*b); v
```

```
Lambda((a, b), a*b)
```



```python
N = v(X, W); N
```

```
Matrix([
[w_11*x_11 + w_21*x_12 + w_31*x_13, w_12*x_11 + w_22*x_12 +
w_32*x_13],
[w_11*x_21 + w_21*x_22 + w_31*x_23, w_12*x_21 + w_22*x_22 +
w_32*x_23],
[w_11*x_31 + w_21*x_32 + w_31*x_33, w_12*x_31 + w_22*x_32 +
w_32*x_33]])
```


