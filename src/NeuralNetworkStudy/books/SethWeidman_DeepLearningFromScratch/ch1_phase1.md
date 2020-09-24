```python
from sympy import Matrix, Symbol, derive_by_array, Lambda, symbols, Derivative, diff
from sympy.abc import x, y, i, j, a, b
```

Defining variable-element matrices $X \in \mathbb{R}^{n \times m}$ and $W \in \mathbb{R}^{m \times p}$:


```python
def var(letter: str, i: int, j: int) -> Symbol:
    letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
    return letter_ij


n,m,p = 3,3,2

X = Matrix(n, m, lambda i,j : var('x', i, j)); X
```




$\displaystyle \left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right]$




```python
W = Matrix(m, p, lambda i,j : var('w', i, j)); W
```




$\displaystyle \left[\begin{matrix}w_{11} & w_{12}\\w_{21} & w_{22}\\w_{31} & w_{32}\end{matrix}\right]$



Defining $N = \nu(X, W) = X \times W$

* $\nu : \mathbb{R}^{(n \times m) \times (m \times p)} \rightarrow \mathbb{R}^{n \times p}$
* $N \in \mathbb{R}^{n \times p}$


```python
v = Lambda((a,b), a*b); v
```




$\displaystyle \left( \left( a, \  b\right) \mapsto a b \right)$




```python
N = v(X, W); N
```




$\displaystyle \left[\begin{matrix}w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} & w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\\w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} & w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\\w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} & w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\end{matrix}\right]$




Defining $S = \sigma_{\text{apply}}(N) = \sigma_{\text{apply}}(\nu(X,W)) = \sigma_\text{apply}(X \times W) = \Big \{ \sigma(XW_{ij}) \Big\}$.


Assume that $\sigma_{\text{apply}} : \mathbb{R}^{n \times p} \rightarrow \mathbb{R}^{n \times p}$ while $\sigma : \mathbb{R} \rightarrow \mathbb{R}$, so the function $\sigma_{\text{apply}}$ takes in a matrix and returns a matrix while the simple $\sigma$ acts on the individual elements $N_{ij} = XW_{ij}$ in the matrix argument $N$ of $\sigma_{\text{apply}}$.

* $\sigma : \mathbb{R} \rightarrow \mathbb{R}$
* $\sigma_\text{apply} : \mathbb{R}^{n \times p} \rightarrow \mathbb{R}^{n \times p}$
* $S \in \mathbb{R}^{n \times p}$


```python
from sympy import Function

# Nvec = Symbol('N', commutative=False)

sigma = Function('sigma')
sigma(N[0,0])
```




$\displaystyle \sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)}$




```python
# way 1 of declaring S
S = N.applyfunc(sigma); S
#type(S)
#Matrix(3, 2, lambda i, j: sigma(N[i,j]))
```




$\displaystyle \left[\begin{matrix}\sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} & \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\\sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} & \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\\sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} & \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right]$




```python
# way 2 of declaring S (better way)
sigmaApply = lambda matrix:  matrix.applyfunc(sigma)

sigmaApply(N)
```




$\displaystyle \left[\begin{matrix}\sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} & \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\\sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} & \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\\sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} & \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right]$




```python
sigmaApply(X**2) # can apply this function to any matrix argument.
```




$\displaystyle \left[\begin{matrix}\sigma{\left(x_{11}^{2} + x_{12} x_{21} + x_{13} x_{31} \right)} & \sigma{\left(x_{11} x_{12} + x_{12} x_{22} + x_{13} x_{32} \right)} & \sigma{\left(x_{11} x_{13} + x_{12} x_{23} + x_{13} x_{33} \right)}\\\sigma{\left(x_{11} x_{21} + x_{21} x_{22} + x_{23} x_{31} \right)} & \sigma{\left(x_{12} x_{21} + x_{22}^{2} + x_{23} x_{32} \right)} & \sigma{\left(x_{13} x_{21} + x_{22} x_{23} + x_{23} x_{33} \right)}\\\sigma{\left(x_{11} x_{31} + x_{21} x_{32} + x_{31} x_{33} \right)} & \sigma{\left(x_{12} x_{31} + x_{22} x_{32} + x_{32} x_{33} \right)} & \sigma{\left(x_{13} x_{31} + x_{23} x_{32} + x_{33}^{2} \right)}\end{matrix}\right]$




```python
S = sigmaApply(v(X,W)) # composing
S
```




$\displaystyle \left[\begin{matrix}\sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} & \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\\sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} & \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\\sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} & \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right]$



Defining $L = \Lambda(S) = \Lambda(\sigma_\text{apply}(\nu(X,W))) = \Lambda \Big(\Big \{ \sigma(XW_{ij}) \Big\} \Big)$. In general, let the function be defined as:
$$
\begin{align}
L &= \Lambda \begin{pmatrix}
   \sigma(XW_{11}) & \sigma(XW_{12}) & ... & \sigma(XW_{1p}) \\
   \sigma(XW_{21}) & \sigma(XW_{22}) & ... & \sigma(XW_{2p}) \\
   \vdots & \vdots & & \vdots \\
   \sigma(XW_{n1}) & \sigma(XW_{n2}) & ... & \sigma(XW_{np})
\end{pmatrix} \\

&= \sum_{i=1}^p \sum_{j = 1}^n  \sigma(XW_{ij}) \\

&= \sigma(XW_{11}) + \sigma{XW_{12}} + ... + \sigma(XW_{np})
\end{align}
$$
* $\Lambda: \mathbb{R}^{n \times p} \rightarrow \mathbb{R}$
* $L \in \mathbb{R}$


```python
lambdaF = lambda matrix : sum(matrix)
lambdaF(S)
```




$\displaystyle \sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} + \sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} + \sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} + \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)} + \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)} + \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}$




```python
L = lambdaF(sigmaApply(v(X, W)))
L
#L = lambda mat1, mat2: lambdaF(sigmaApply(v(mat1, mat2)))
#L(X, W)
```




$\displaystyle \sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} + \sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} + \sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} + \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)} + \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)} + \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}$






```python
#derive_by_array(L, X)
```


```python
derive_by_array(L, S)
```




$\displaystyle \left[\begin{matrix}1 & 1\\1 & 1\\1 & 1\end{matrix}\right]$




```python
from sympy import sympify, lambdify
n = lambdify((X[0,0],X[0,1],X[0,2],W[0,0],W[1,0],W[2,0]), N[0,0])
n(1,2,3,4,3,2)
```




    16




```python
f = Function('f') #(sympify(N[0,0]))
f(N[0,0])
```




$\displaystyle f{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)}$




```python
f(N[0,0]).diff(X[0,0])



```




$\displaystyle w_{11} \left. \frac{d}{d \xi_{1}} f{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }}$




```python
n = v(X,W); n
n11 = Function('{}'.format(n[0,0]))
n11
```




    w_11*x_11 + w_21*x_12 + w_31*x_13




```python
s_ij = Function('s_ij')
sig = Function('sig')(x)
```


```python

# KEY: got not expecting UndefinedFunction error again here too
#S_ij = Matrix(3, 2, lambda i,j: Function('s_{}{}'.format(i+1,j+1))(Function('{}'.format(N[i,j]))))


```


```python
#S_ij[0,0](sympify(N[0,0])).diff(sympify(N[0,0]))
F = 3*x*y

xy = Symbol('{}'.format(F))
xy.subs({x:3})
sympify(xy).subs({x:3})
```




$\displaystyle 3*x*y$



Sympy Example of trying to differentiate with respect to an **expression** not just a variable.


```python
from sympy.abc import t

F = Function('F')
f = Function('f')
U = f(t)
V = U.diff(t)

direct = F(t, U, V).diff(U); direct
```




$\displaystyle \left. \frac{\partial}{\partial \xi_{2}} F{\left(t,\xi_{2},\frac{d}{d t} f{\left(t \right)} \right)} \right|_{\substack{ \xi_{2}=f{\left(t \right)} }}$




```python
F(t,U,V)
```




$\displaystyle F{\left(t,f{\left(t \right)},\frac{d}{d t} f{\left(t \right)} \right)}$




```python
F(t,U,V).subs(U,x)
```




$\displaystyle F{\left(t,x,\frac{d}{d t} x \right)}$




```python
F(t,U,V).subs(U,x).diff(x)
```




$\displaystyle \left. \frac{\partial}{\partial \xi_{2}} F{\left(t,\xi_{2},\frac{d}{d t} x \right)} \right|_{\substack{ \xi_{2}=x }}$




```python
F(t,U,V).subs(U,x).diff(x).subs(x, U)
```




$\displaystyle \left. \frac{\partial}{\partial \xi_{2}} F{\left(t,\xi_{2},\frac{d}{d t} f{\left(t \right)} \right)} \right|_{\substack{ \xi_{2}=f{\left(t \right)} }}$




```python
indirect = F(t,U,V).subs(U, x).diff(x).subs(x,U); indirect
```




$\displaystyle \left. \frac{\partial}{\partial \xi_{2}} F{\left(t,\xi_{2},\frac{d}{d t} f{\left(t \right)} \right)} \right|_{\substack{ \xi_{2}=f{\left(t \right)} }}$




```python
F = Lambda((x,y), 3*x* y)
F(1,2)
```




$\displaystyle 6$




```python
U = x*y
G = 3*x*y
xy
```




$\displaystyle 3*x*y$




```python
F.diff(xy)
```




$\displaystyle 0$




```python
# derive_by_array(S, N) # ERROR
```


```python
s11 = S[0,0]
s11
```




$\displaystyle \sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)}$




```python

#s11.diff(n11)
```


```python
derive_by_array(L, S)
```




$\displaystyle \left[\begin{matrix}1 & 1\\1 & 1\\1 & 1\end{matrix}\right]$




```python

x, y, r, t = symbols('x y r t') # r (radius), t (angle theta)
f, g, h = symbols('f g h', cls=Function)
h = g(f(x))
Derivative(h, f(x)).doit()



```




$\displaystyle \frac{d}{d f{\left(x \right)}} g{\left(f{\left(x \right)} \right)}$




```python
h.args[0]
h.diff(h.args[0])
```




$\displaystyle \frac{d}{d f{\left(x \right)}} g{\left(f{\left(x \right)} \right)}$




```python
S = sigmaApply(v(X,W)); S
```




$\displaystyle \left[\begin{matrix}\sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} & \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\\sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} & \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\\sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} & \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right]$




```python
from sympy.abc import n

n11 = (X*W)[0,0]
m = lambda mat1, mat2: sympify(Symbol('{}'.format((mat1 * mat2)[0,0] )))
s = sigma(m(X,W)); s
```




$\displaystyle \sigma{\left(w_{11*x 11 + w 21*x 12 + w 31*x 13} \right)}$




```python
s.subs({W[0,0]: 14}) # doesn't work to substitute into an undefined function
```




$\displaystyle \sigma{\left(w_{11*x 11 + w 21*x 12 + w 31*x 13} \right)}$




```python
Derivative(s, m(X,W)).doit()
```




$\displaystyle \frac{d}{d w_{11*x 11 + w 21*x 12 + w 31*x 13}} \sigma{\left(w_{11*x 11 + w 21*x 12 + w 31*x 13} \right)}$




```python

#s11 = Function('s_{11}')(n11); s11
#sigma(n11).diff(n11)

#s11.diff(n11)
sigma(n11)
```




$\displaystyle \sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)}$




```python
# ERROR HERE TOO
type(sigma(n11).args[0])
```




    sympy.core.add.Add




```python
type(n11)
```




    sympy.core.add.Add




```python
#sigma(n11).diff(sigma(n11).args[0]) ## ERROR
```


```python

```


```python
b = Symbol('{}'.format(n11))
ns_11 = Function(b, real=True)
ns_11


# ERROR cannot diff wi.r. to undefinedfunction
# sigma(n11).diff(ns_11)


#
#sigma(b).diff(b).subs({b:1})
```




    w_11*x_11 + w_21*x_12 + w_31*x_13




```python
f, g = symbols('f g', cls=Function)
xy = Symbol('x*y'); xy
#sympify(xy).subs({x:2, y:4})
f(g(x,y)).diff(xy)
```




$\displaystyle 0$




```python
# TODO SEEM to have got the expression but it is not working since can't substitute anything .... ???
f(xy).diff(xy).subs({x:2})
```




$\displaystyle \frac{d}{d x*y} f{\left(x*y \right)}$




```python
Function("x*y")(x,y)
xyf = lambdify([x,y],xy)
xyf(3,4)
f(g(xy)).diff(xy)
#
```




$\displaystyle \frac{d}{d g{\left(x*y \right)}} f{\left(g{\left(x*y \right)} \right)} \frac{d}{d x*y} g{\left(x*y \right)}$




```python
xyd = Derivative(x*y, x*y,0).doit();xyd

#Derivative(3*xyd, xyd, 1).doit() ### ERROR can't calc deriv w.r.t to x*y
```




$\displaystyle x y$




```python
#derive_by_array(S, N)




```


```python

```


```python

```
