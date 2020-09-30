---
jupyter:
  jupytext:
    cell_metadata_filter: title,-all
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: pymatrix_env
    language: python
    name: pymatrix_env
---

```python title="codecell"
from sympy import Matrix, Symbol, derive_by_array, Lambda, symbols, Derivative, diff
from sympy.abc import x, y, i, j, a, b
```

<!-- #region markdown -->

Defining variable-element matrices $X \in \mathbb{R}^{n \times m}$ and $W \in \mathbb{R}^{m \times p}$:
<!-- #endregion -->
```python title="codecell"
def var(letter: str, i: int, j: int) -> Symbol:
    letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
    return letter_ij


n,m,p = 3,3,2

X = Matrix(n, m, lambda i,j : var('x', i, j)); X
```
```python title="codecell"
W = Matrix(m, p, lambda i,j : var('w', i, j)); W
```

<!-- #region markdown -->
Defining $N = \nu(X, W) = X \times W$

* $\nu : \mathbb{R}^{(n \times m) \times (m \times p)} \rightarrow \mathbb{R}^{n \times p}$
* $N \in \mathbb{R}^{n \times p}$
<!-- #endregion -->
```python title="codecell"
v = Lambda((a,b), a*b); v
```

```python title="codecell"
N = v(X, W); N
```

<!-- #region markdown -->

Defining $S = \sigma_{\text{apply}}(N) = \sigma_{\text{apply}}(\nu(X,W)) = \sigma_\text{apply}(X \times W) = \Big \{ \sigma(XW_{ij}) \Big\}$.


Assume that $\sigma_{\text{apply}} : \mathbb{R}^{n \times p} \rightarrow \mathbb{R}^{n \times p}$ while $\sigma : \mathbb{R} \rightarrow \mathbb{R}$, so the function $\sigma_{\text{apply}}$ takes in a matrix and returns a matrix while the simple $\sigma$ acts on the individual elements $N_{ij} = XW_{ij}$ in the matrix argument $N$ of $\sigma_{\text{apply}}$.

* $\sigma : \mathbb{R} \rightarrow \mathbb{R}$
* $\sigma_\text{apply} : \mathbb{R}^{n \times p} \rightarrow \mathbb{R}^{n \times p}$
* $S \in \mathbb{R}^{n \times p}$
<!-- #endregion -->
```python title="codecell"
from sympy import Function

# Nvec = Symbol('N', commutative=False)

sigma = Function('sigma')
sigma(N[0,0])
```

```python title="codecell"
# way 1 of declaring S
S = N.applyfunc(sigma); S
#type(S)
#Matrix(3, 2, lambda i, j: sigma(N[i,j]))
```
```python title="codecell"
# way 2 of declaring S (better way)
sigmaApply = lambda matrix:  matrix.applyfunc(sigma)

sigmaApply(N)
```

```python title="codecell"
sigmaApply(X**2) # can apply this function to any matrix argument.
```
```python title="codecell"
S = sigmaApply(v(X,W)) # composing
S
```

<!-- #region markdown -->
Defining $L = \Lambda(S) = \Lambda(\sigma_\text{apply}(\nu(X,W))) = \Lambda \Big(\Big \{ \sigma(XW_{ij}) \Big\} \Big)$. In general, let the function be defined as:

$\begin{aligned}
L &= \Lambda \begin{pmatrix}
   \sigma(XW_{11}) & \sigma(XW_{12}) & ... & \sigma(XW_{1p}) \\
   \sigma(XW_{21}) & \sigma(XW_{22}) & ... & \sigma(XW_{2p}) \\
   \vdots & \vdots & & \vdots \\
   \sigma(XW_{n1}) & \sigma(XW_{n2}) & ... & \sigma(XW_{np})
\end{pmatrix} \\
&= \sum_{i=1}^p \sum_{j = 1}^n  \sigma(XW_{ij}) \\
&= \sigma(XW_{11}) + \sigma{XW_{12}} + ... + \sigma(XW_{np})
\end{aligned}$

NOTE HERE:
* $\Lambda: \mathbb{R}^{n \times p} \rightarrow \mathbb{R}$
* $L \in \mathbb{R}$
<!-- #endregion -->
```python title="codecell"
lambdaF = lambda matrix : sum(matrix)
lambdaF(S)
```
```python title="codecell"
L = lambdaF(sigmaApply(v(X, W)))
L
#L = lambda mat1, mat2: lambdaF(sigmaApply(v(mat1, mat2)))
#L(X, W)
```


<!-- #region markdown -->

<!-- #endregion -->
```python title="codecell"
#derive_by_array(L, X)
```
```python title="codecell"
derive_by_array(L, S)
```
```python title="codecell"
from sympy import sympify, lambdify
n = lambdify((X[0,0],X[0,1],X[0,2],W[0,0],W[1,0],W[2,0]), N[0,0])
n(1,2,3,4,3,2)
```

```python title="codecell"
f = Function('f') #(sympify(N[0,0]))
f(N[0,0])
```
```python title="codecell"
f(N[0,0]).diff(X[0,0])



```

```python title="codecell"
n = v(X,W); n
n11 = Function('{}'.format(n[0,0]))
n11
```
```python title="codecell"
s_ij = Function('s_ij')
sig = Function('sig')(x)
```


```python title="codecell"

# KEY: got not expecting UndefinedFunction error again here too
#S_ij = Matrix(3, 2, lambda i,j: Function('s_{}{}'.format(i+1,j+1))(Function('{}'.format(N[i,j]))))


```

```python title="codecell"
#S_ij[0,0](sympify(N[0,0])).diff(sympify(N[0,0]))
F = 3*x*y

xy = Symbol('{}'.format(F))
xy.subs({x:3})
sympify(xy).subs({x:3})
```

<!-- #region markdown -->
Sympy Example of trying to differentiate with respect to an **expression** not just a variable.
<!-- #endregion -->
```python title="codecell"
from sympy.abc import t

F = Function('F')
f = Function('f')
U = f(t)
V = U.diff(t)

direct = F(t, U, V).diff(U); direct
```

```python title="codecell"
F(t,U,V)
```
```python title="codecell"
F(t,U,V).subs(U,x)
```
```python title="codecell"
F(t,U,V).subs(U,x).diff(x)
```
```python title="codecell"
F(t,U,V).subs(U,x).diff(x).subs(x, U)
```
```python title="codecell"
indirect = F(t,U,V).subs(U, x).diff(x).subs(x,U); indirect
```
```python title="codecell"
F = Lambda((x,y), 3*x* y)
F(1,2)
```
```python title="codecell"
U = x*y
G = 3*x*y
xy
```
```python title="codecell"
F.diff(xy)
```
```python title="codecell"
# derive_by_array(S, N) # ERROR
```

```python title="codecell"
s11 = S[0,0]
s11
```
```python title="codecell"

#s11.diff(n11)
```

```python title="codecell"
derive_by_array(L, S)
```


```python title="codecell"

x, y, r, t = symbols('x y r t') # r (radius), t (angle theta)
f, g, h = symbols('f g h', cls=Function)
h = g(f(x))
Derivative(h, f(x)).doit()



```

```python title="codecell"
h.args[0]
h.diff(h.args[0])
```


```python title="codecell"
S = sigmaApply(v(X,W)); S
```

```python title="codecell"
from sympy.abc import n

n11 = (X*W)[0,0]
m = lambda mat1, mat2: sympify(Symbol('{}'.format((mat1 * mat2)[0,0] )))
s = sigma(m(X,W)); s
```
```python title="codecell"
s.subs({W[0,0]: 14}) # doesn't work to substitute into an undefined function
```
```python title="codecell"
Derivative(s, m(X,W)).doit()
```
```python title="codecell"

#s11 = Function('s_{11}')(n11); s11
#sigma(n11).diff(n11)

#s11.diff(n11)
sigma(n11)
```
```python title="codecell"
# ERROR HERE TOO
type(sigma(n11).args[0])
```
```python title="codecell"
type(n11)
```
```python title="codecell"
#sigma(n11).diff(sigma(n11).args[0]) ## ERROR
```
```python title="codecell"

```

```python title="codecell"
b = Symbol('{}'.format(n11))
ns_11 = Function(b, real=True)
ns_11


# ERROR cannot diff wi.r. to undefinedfunction
# sigma(n11).diff(ns_11)


#
#sigma(b).diff(b).subs({b:1})
```


```python title="codecell"
f, g = symbols('f g', cls=Function)
xy = Symbol('x*y'); xy
#sympify(xy).subs({x:2, y:4})
f(g(x,y)).diff(xy)
```
```python title="codecell"
# TODO SEEM to have got the expression but it is not working since can't substitute anything .... ???
f(xy).diff(xy).subs({x:2})
```
```python title="codecell"
Function("x*y")(x,y)
xyf = lambdify([x,y],xy)
xyf(3,4)
f(g(xy)).diff(xy)
#
```

```python title="codecell"
xyd = Derivative(x*y, x*y,0).doit();xyd

#Derivative(3*xyd, xyd, 1).doit() ### ERROR can't calc deriv w.r.t to x*y
```

```python title="codecell"
#derive_by_array(S, N)




```

```python title="codecell"

```
```python title="codecell"

```
