```python
from sympy.interactive import init_printing
from sympy import Matrix, Symbol, derive_by_array, Lambda, Function, MatrixSymbol, Derivative, diff, symbols
from sympy import var
from sympy.abc import x, i, j, a, b


init_printing(pretty_print=True, wrap_line=True, num_columns=60)
```


```python
def myvar(letter: str, i: int, j: int) -> Symbol:
    letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
    return letter_ij


n,m,p = 3,3,2

X = Matrix(n, m, lambda i,j : myvar('x', i, j)); X
```




$\displaystyle \left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right]$




```python
W = Matrix(m, p, lambda i,j : myvar('w', i, j)); W
```




$\displaystyle \left[\begin{matrix}w_{11} & w_{12}\\w_{21} & w_{22}\\w_{31} & w_{32}\end{matrix}\right]$




```python
A = MatrixSymbol('X',3,3); Matrix(A)
B = MatrixSymbol('W',3,2)
```


```python

```


```python

```


```python
v = lambda a,b: a*b

vL = Lambda((a,b), a*b)

n = Function('v') #, Lambda((a,b), a*b))

vN = lambda mat1, mat2: Matrix(mat1.shape[0], mat2.shape[1], lambda i, j: Symbol("n_{}{}".format(i+1, j+1))); vN

Nelem = vN(X, W); Nelem
```




$\displaystyle \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right]$




```python
n(X,W)
```




$\displaystyle v{\left(\left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right],\left[\begin{matrix}w_{11} & w_{12}\\w_{21} & w_{22}\\w_{31} & w_{32}\end{matrix}\right] \right)}$




```python
n(A,B)
```




$\displaystyle v{\left(X,W \right)}$




```python
n(X,W).replace(n, v) # replace works when v = python lambda
```




$\displaystyle \left[\begin{matrix}w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} & w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\\w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} & w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\\w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} & w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\end{matrix}\right]$




```python
n(X,W).subs({n: vL}) # subs works when v = sympy lambda
```




$\displaystyle \left[\begin{matrix}w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} & w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\\w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} & w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\\w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} & w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\end{matrix}\right]$




```python
n(X,W).replace(n, vL)
```




$\displaystyle \left[\begin{matrix}w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} & w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\\w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} & w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\\w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} & w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\end{matrix}\right]$




```python
n(X,W).subs({n: v})# subs() doesn't work when v is python lambda
```




$\displaystyle v{\left(\left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right],\left[\begin{matrix}w_{11} & w_{12}\\w_{21} & w_{22}\\w_{31} & w_{32}\end{matrix}\right] \right)}$




```python
Matrix(n(A,B).subs({n: vL}))
```




$\displaystyle \left[\begin{array}{cc}W_{0, 0} X_{0, 0} + W_{1, 0} X_{0, 1} + W_{2, 0} X_{0, 2} & W_{0, 1} X_{0, 0} + W_{1, 1} X_{0, 1} + W_{2, 1} X_{0, 2}\\W_{0, 0} X_{1, 0} + W_{1, 0} X_{1, 1} + W_{2, 0} X_{1, 2} & W_{0, 1} X_{1, 0} + W_{1, 1} X_{1, 1} + W_{2, 1} X_{1, 2}\\W_{0, 0} X_{2, 0} + W_{1, 0} X_{2, 1} + W_{2, 0} X_{2, 2} & W_{0, 1} X_{2, 0} + W_{1, 1} X_{2, 1} + W_{2, 1} X_{2, 2}\end{array}\right]$




```python
#N = v(X, W); N
N = n(A,B); N
```




$\displaystyle v{\left(X,W \right)}$




```python
N.replace(n, v)
```




$\displaystyle X W$




```python
N.replace(n, v).subs({A: X, B:W}) # replacing ariable values after doing function doesn't make the function apply directly on the values (matrices), need to replace values before the function is replaced, so that the function can act on them while they are given/alive.
```




$\displaystyle \left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right] \left[\begin{matrix}w_{11} & w_{12}\\w_{21} & w_{22}\\w_{31} & w_{32}\end{matrix}\right]$




```python
N.subs({n: vL, A:X, B:W})
```




$\displaystyle \left[\begin{matrix}w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} & w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\\w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} & w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\\w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} & w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\end{matrix}\right]$




```python
Nspec = N.subs({A:X, B:W}).replace(n, v); Nspec
```




$\displaystyle \left[\begin{matrix}w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} & w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\\w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} & w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\\w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} & w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\end{matrix}\right]$




```python

```


```python
N.diff(N)
```




$\displaystyle 1$




```python
N.diff(X)
```




$\displaystyle \left[\begin{matrix}0 & 0 & 0\\0 & 0 & 0\\0 & 0 & 0\end{matrix}\right]$




```python

```


```python



```


```python
# way 2 of declaring S (better way)
sigma = Function('sigma')

sigmaApply = Function("sigma_apply") #lambda matrix:  matrix.applyfunc(sigma)

sigmaApply_ = lambda matrix: matrix.applyfunc(sigma)

sigmaApply(A)
```




$\displaystyle \sigma_{apply}{\left(X \right)}$




```python
sigmaApply(A).subs({A: X})
```




$\displaystyle \sigma_{apply}{\left(\left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right] \right)}$




```python
sigmaApply_(A)
```




$\displaystyle {\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({X}\right)$




```python
sigmaApply(A).subs({A: X}).replace(sigmaApply, sigmaApply_) # NOTE: subs of functions doesn't work, replace actually evaluates the replaced function!
```




$\displaystyle \left[\begin{matrix}\sigma{\left(x_{11} \right)} & \sigma{\left(x_{12} \right)} & \sigma{\left(x_{13} \right)}\\\sigma{\left(x_{21} \right)} & \sigma{\left(x_{22} \right)} & \sigma{\left(x_{23} \right)}\\\sigma{\left(x_{31} \right)} & \sigma{\left(x_{32} \right)} & \sigma{\left(x_{33} \right)}\end{matrix}\right]$




```python
S = sigmaApply(N); S
```




$\displaystyle \sigma_{apply}{\left(v{\left(X,W \right)} \right)}$




```python
Derivative(S, S)
```




$\displaystyle \frac{\partial}{\partial \sigma_{apply}{\left(v{\left(X,W \right)} \right)}} \sigma_{apply}{\left(v{\left(X,W \right)} \right)}$




```python
Derivative(S, S).doit()
```




$\displaystyle 1$




```python
Derivative(S, n(A,B)).doit()
```




$\displaystyle \frac{\partial}{\partial v{\left(X,W \right)}} \sigma_{apply}{\left(v{\left(X,W \right)} \right)}$




```python
#lambd = Function("lambda")
#Lagain = lambd(sigmaApply(n(A))); Lagain



# diff(Lagain, A) # never execute
#
```


```python
S.replace(A,X).replace(B,W)
```




$\displaystyle \sigma_{apply}{\left(v{\left(\left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right],\left[\begin{matrix}w_{11} & w_{12}\\w_{21} & w_{22}\\w_{31} & w_{32}\end{matrix}\right] \right)} \right)}$




```python
S.replace(n, v)
```




$\displaystyle \sigma_{apply}{\left(X W \right)}$




```python
S.subs({A:X, B:W}).replace(n, v)
```




$\displaystyle \sigma_{apply}{\left(\left[\begin{matrix}w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} & w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\\w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} & w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\\w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} & w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\end{matrix}\right] \right)}$




```python
Sspec = S.subs({A:X, B:W}).replace(n, v).replace(sigmaApply, sigmaApply_)
Sspec
```




$\displaystyle \left[\begin{matrix}\sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} & \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\\sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} & \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\\sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} & \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right]$




```python
S.replace(n, vN) #.replace(sigmaApply, sigmaApply_)
```




$\displaystyle \sigma_{apply}{\left(\left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right] \right)}$




```python
Selem = S.replace(n, vN).replace(sigmaApply, sigmaApply_); Selem
```




$\displaystyle \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]$




```python
import itertools

elemToSpecD = dict(itertools.chain(*[[(Nelem[i, j], Nspec[i, j]) for j in range(2)] for i in range(3)]))

elemToSpec = list(elemToSpecD.items())

Matrix(elemToSpec)
```




$\displaystyle \left[\begin{matrix}n_{11} & w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13}\\n_{12} & w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\\n_{21} & w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23}\\n_{22} & w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\\n_{31} & w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33}\\n_{32} & w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\end{matrix}\right]$




```python
elemToSpecFuncD = dict(itertools.chain(*[[(Nelem[i, j], Function("n_{}{}".format(i + 1, j + 1))(Nspec[i, j])) for j in range(2)] for i in range(3)]))

elemToSpecFunc = list(elemToSpecFuncD.items())

Matrix(elemToSpecFunc)
```




$\displaystyle \left[\begin{matrix}n_{11} & \operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)}\\n_{12} & \operatorname{n_{12}}{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\n_{21} & \operatorname{n_{21}}{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)}\\n_{22} & \operatorname{n_{22}}{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\n_{31} & \operatorname{n_{31}}{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)}\\n_{32} & \operatorname{n_{32}}{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right]$




```python
elemToSpecFuncArgsD = dict(itertools.chain(*[[(Nelem[i, j], Function("n_{}{}".format(i + 1, j + 1))(*X,*W)) for j in range(2)] for i in range(3)]))

elemToSpecFuncArgs = list(elemToSpecFuncArgsD.items())

Matrix(elemToSpecFuncArgs)
```




$\displaystyle \left[\begin{matrix}n_{11} & \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\n_{12} & \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\n_{21} & \operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\n_{22} & \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\n_{31} & \operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\n_{32} & \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right]$




```python
Selem
```




$\displaystyle \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]$




```python
Selem.subs(elemToSpecD)
```




$\displaystyle \left[\begin{matrix}\sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} & \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\\sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} & \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\\sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} & \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right]$




```python
Selem[0,1].diff(Nelem[0,1])
```




$\displaystyle \frac{d}{d n_{12}} \sigma{\left(n_{12} \right)}$




```python
Selem[0,1].diff(Nelem[0,1]).subs({Nelem[0,1] : Nspec[0,1]})
#Selem[0,1].diff(Nelem[0,1]).subs(dict([{Nelem[0,1] : Nspec[0,1]}]))
```




$\displaystyle \left. \frac{d}{d n_{12}} \sigma{\left(n_{12} \right)} \right|_{\substack{ n_{12}=w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} }}$




```python
Selem[0,1].diff(Nelem[0,1]).subs({Nelem[0,1] : Nspec[0,1]}).subs({Nspec[0,1] : 23})
```




$\displaystyle \left. \frac{d}{d n_{12}} \sigma{\left(n_{12} \right)} \right|_{\substack{ n_{12}=23 }}$




```python
Selem[0,1].diff(Nelem[0,1]).subs({Nelem[0,1] : Nspec[0,1]}).replace(sigma, lambda x: 8*x**3)
```




$\displaystyle \left. \frac{d}{d n_{12}} 8 n_{12}^{3} \right|_{\substack{ n_{12}=w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} }}$




```python
Selem[0,1].diff(Nelem[0,1]).replace(sigma, lambda x: 8*x**3)
```




$\displaystyle \frac{d}{d n_{12}} 8 n_{12}^{3}$




```python
Selem[0,1].diff(Nelem[0,1]).replace(sigma, lambda x: 8*x**3).doit()
```




$\displaystyle 24 n_{12}^{2}$




```python
# ### GOT IT: can replace now with expression and do derivative with respect to that expression.
Selem[0,1].diff(Nelem[0,1]).subs({Nelem[0,1] : Nspec[0,1]}).replace(sigma, lambda x: 8*x**3).doit()
```




$\displaystyle 24 \left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\right)^{2}$




```python
Selem[0,1].subs({Nelem[0,1] : Nspec[0,1]}).diff(X[0,1])#.subs({Nelem[0,1] : Nspec[0,1]})
```




$\displaystyle w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} }}$




```python
Selem
```




$\displaystyle \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]$




```python
nt = Nelem.subs(elemToSpecFunc); nt
```




$\displaystyle \left[\begin{matrix}\operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} & \operatorname{n_{12}}{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\\operatorname{n_{21}}{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} & \operatorname{n_{22}}{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\\operatorname{n_{31}}{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} & \operatorname{n_{32}}{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right]$




```python
st = Selem.subs(elemToSpecFunc); st
```




$\displaystyle \left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)} \right)}\end{matrix}\right]$




```python
st.diff(nt)
```




$\displaystyle \left[\begin{matrix}\left[\begin{matrix}\frac{\partial}{\partial \operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)}} \sigma{\left(\operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} \right)} & 0\\0 & 0\\0 & 0\end{matrix}\right] & \left[\begin{matrix}0 & \frac{\partial}{\partial \operatorname{n_{12}}{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}} \sigma{\left(\operatorname{n_{12}}{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)} \right)}\\0 & 0\\0 & 0\end{matrix}\right]\\\left[\begin{matrix}0 & 0\\\frac{\partial}{\partial \operatorname{n_{21}}{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)}} \sigma{\left(\operatorname{n_{21}}{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} \right)} & 0\\0 & 0\end{matrix}\right] & \left[\begin{matrix}0 & 0\\0 & \frac{\partial}{\partial \operatorname{n_{22}}{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}} \sigma{\left(\operatorname{n_{22}}{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)} \right)}\\0 & 0\end{matrix}\right]\\\left[\begin{matrix}0 & 0\\0 & 0\\\frac{\partial}{\partial \operatorname{n_{31}}{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)}} \sigma{\left(\operatorname{n_{31}}{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} \right)} & 0\end{matrix}\right] & \left[\begin{matrix}0 & 0\\0 & 0\\0 & \frac{\partial}{\partial \operatorname{n_{32}}{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}} \sigma{\left(\operatorname{n_{32}}{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)} \right)}\end{matrix}\right]\end{matrix}\right]$




```python
st[0,0].diff(st[0,0].args[0])
```




$\displaystyle \frac{\partial}{\partial \operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)}} \sigma{\left(\operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} \right)}$




```python
st[0,0].diff(X[0,0])
```




$\displaystyle w_{11} \frac{\partial}{\partial \operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)}} \sigma{\left(\operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} \right)} \left. \frac{d}{d \xi_{1}} \operatorname{n_{11}}{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }}$




```python
st[0,0].diff(st[1,0].args[0])
```




$\displaystyle 0$




```python
Selem.diff(Nelem)
```




$\displaystyle \left[\begin{matrix}\left[\begin{matrix}\frac{d}{d n_{11}} \sigma{\left(n_{11} \right)} & 0\\0 & 0\\0 & 0\end{matrix}\right] & \left[\begin{matrix}0 & \frac{d}{d n_{12}} \sigma{\left(n_{12} \right)}\\0 & 0\\0 & 0\end{matrix}\right]\\\left[\begin{matrix}0 & 0\\\frac{d}{d n_{21}} \sigma{\left(n_{21} \right)} & 0\\0 & 0\end{matrix}\right] & \left[\begin{matrix}0 & 0\\0 & \frac{d}{d n_{22}} \sigma{\left(n_{22} \right)}\\0 & 0\end{matrix}\right]\\\left[\begin{matrix}0 & 0\\0 & 0\\\frac{d}{d n_{31}} \sigma{\left(n_{31} \right)} & 0\end{matrix}\right] & \left[\begin{matrix}0 & 0\\0 & 0\\0 & \frac{d}{d n_{32}} \sigma{\left(n_{32} \right)}\end{matrix}\right]\end{matrix}\right]$




```python
Selem.diff(Nelem).subs(elemToSpecFunc)
```




$\displaystyle \left[\begin{matrix}\left[\begin{matrix}\frac{\partial}{\partial \operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)}} \sigma{\left(\operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} \right)} & 0\\0 & 0\\0 & 0\end{matrix}\right] & \left[\begin{matrix}0 & \frac{\partial}{\partial \operatorname{n_{12}}{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}} \sigma{\left(\operatorname{n_{12}}{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)} \right)}\\0 & 0\\0 & 0\end{matrix}\right]\\\left[\begin{matrix}0 & 0\\\frac{\partial}{\partial \operatorname{n_{21}}{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)}} \sigma{\left(\operatorname{n_{21}}{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} \right)} & 0\\0 & 0\end{matrix}\right] & \left[\begin{matrix}0 & 0\\0 & \frac{\partial}{\partial \operatorname{n_{22}}{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}} \sigma{\left(\operatorname{n_{22}}{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)} \right)}\\0 & 0\end{matrix}\right]\\\left[\begin{matrix}0 & 0\\0 & 0\\\frac{\partial}{\partial \operatorname{n_{31}}{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)}} \sigma{\left(\operatorname{n_{31}}{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} \right)} & 0\end{matrix}\right] & \left[\begin{matrix}0 & 0\\0 & 0\\0 & \frac{\partial}{\partial \operatorname{n_{32}}{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}} \sigma{\left(\operatorname{n_{32}}{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)} \right)}\end{matrix}\right]\end{matrix}\right]$




```python
# CAN even replace elements after have done an operation on them!!! replacing n_21 * 2 with the number 4.
Sspec.subs({Nspec[0, 0]: 3}).replace(sigma, lambda x: 2 * x).replace(Nspec[2, 1] * 2, 4)



```




$\displaystyle \left[\begin{matrix}6 & 2 w_{12} x_{11} + 2 w_{22} x_{12} + 2 w_{32} x_{13}\\2 w_{11} x_{21} + 2 w_{21} x_{22} + 2 w_{31} x_{23} & 2 w_{12} x_{21} + 2 w_{22} x_{22} + 2 w_{32} x_{23}\\2 w_{11} x_{31} + 2 w_{21} x_{32} + 2 w_{31} x_{33} & 4\end{matrix}\right]$




```python
lambd = Function("lambda")
lambd_ = lambda matrix : sum(matrix)


vN(X, W)
```




$\displaystyle \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right]$




```python
vN(A, B)
```




$\displaystyle \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right]$




```python
L = lambd(S); L
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)}$




```python
Nelem
```




$\displaystyle \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right]$




```python
L.replace(n, vN)
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(\left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right] \right)} \right)}$




```python
L.replace(n, vN).replace(sigmaApply, sigmaApply_)
```




$\displaystyle \lambda{\left(\left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] \right)}$




```python
L.replace(n, v)
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(X W \right)} \right)}$




```python

L.replace(n, v).replace(sigmaApply, sigmaApply_)
```




$\displaystyle \lambda{\left({\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right) \right)}$




```python
L.subs({A:X, B:W}).replace(n, vL).replace(sigmaApply, sigmaApply_)
```




$\displaystyle \lambda{\left(\left[\begin{matrix}\sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} & \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\\sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} & \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\\sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} & \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right] \right)}$




```python
L.replace(n, vN)
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(\left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right] \right)} \right)}$




```python
L.replace(n, vN).subs({A:X, B:W}).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)






```




$\displaystyle \sigma{\left(n_{11} \right)} + \sigma{\left(n_{12} \right)} + \sigma{\left(n_{21} \right)} + \sigma{\left(n_{22} \right)} + \sigma{\left(n_{31} \right)} + \sigma{\left(n_{32} \right)}$




```python
from sympy import symbols, Derivative

x, y, r, t = symbols('x y r t') # r (radius), t (angle theta)
f, g, h = symbols('f g h', cls=Function)
h = g(f(x))

Derivative(h, f(x)).doit()
```




$\displaystyle \frac{d}{d f{\left(x \right)}} g{\left(f{\left(x \right)} \right)}$




```python
# Never do this gives recursion ERROR (max depth exceeded)
# h = g(f(A))
# Derivative(h, A).doit()
```


```python

```


```python
from sympy.abc import a, b

Llower = lambd(sigmaApply(n(a, b)))
Llower
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(v{\left(a,b \right)} \right)} \right)}$




```python
Derivative(Llower, a).doit()
```




$\displaystyle \frac{\partial}{\partial \sigma_{apply}{\left(v{\left(a,b \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(v{\left(a,b \right)} \right)} \right)} \frac{\partial}{\partial v{\left(a,b \right)}} \sigma_{apply}{\left(v{\left(a,b \right)} \right)} \frac{\partial}{\partial a} v{\left(a,b \right)}$




```python

```


```python
# ### WAY 1: of substituting to differentiate with respect to expression:
n_ij = Function('n_ij')
n_ij(A,B) # (N[0,0]); n_ij
```




$\displaystyle \operatorname{n_{ij}}{\left(X,W \right)}$




```python
n_ij(A,B).args
```




$\displaystyle \left( X, \  W\right)$




```python
# sigma(n_ij).diff(n_ij).replace(n_ij, N[0,0]) # ERROR cannot deriv wi.r.t to the expression w11*x11 + ...

sigma(n_ij(A,B)).diff(n_ij(A,B))
```




$\displaystyle \frac{\partial}{\partial \operatorname{n_{ij}}{\left(X,W \right)}} \sigma{\left(\operatorname{n_{ij}}{\left(X,W \right)} \right)}$




```python
sigma(n_ij(*X,*W)).diff(X[0,0])
```




$\displaystyle \frac{\partial}{\partial x_{11}} \operatorname{n_{ij}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \frac{\partial}{\partial \operatorname{n_{ij}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}} \sigma{\left(\operatorname{n_{ij}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)}$




```python
nab_ij = n_ij(A,B)
sigma(nab_ij).diff(nab_ij)#.subs({nab_ij : Nspec[0, 0]})
```




$\displaystyle \frac{\partial}{\partial \operatorname{n_{ij}}{\left(X,W \right)}} \sigma{\left(\operatorname{n_{ij}}{\left(X,W \right)} \right)}$




```python
sigma(nab_ij).diff(nab_ij).subs({nab_ij : Nspec[2, 1]})
```




$\displaystyle \left. \frac{d}{d \xi} \sigma{\left(\xi \right)} \right|_{\substack{ \xi=w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} }}$




```python
sigma(nab_ij).diff(nab_ij).subs({nab_ij : Nspec[2,1]}).subs({X[2,1]:77777})
```




$\displaystyle \left. \frac{d}{d \xi} \sigma{\left(\xi \right)} \right|_{\substack{ \xi=w_{12} x_{31} + 77777 w_{22} + w_{32} x_{33} }}$




```python
sigma(nab_ij).diff(nab_ij).subs({nab_ij : 23}) # ERROR if using replace() since it says can't calc derivs w.r.t to the x_11*w_11 + ...
```




$\displaystyle \left. \frac{d}{d \xi} \sigma{\left(\xi \right)} \right|_{\substack{ \xi=23 }}$




```python
sigma(nab_ij).diff(nab_ij).subs({nab_ij : Nspec[2,1]}).doit()
```




$\displaystyle \left. \frac{d}{d \xi} \sigma{\left(\xi \right)} \right|_{\substack{ \xi=w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} }}$




```python
sigma(nab_ij).subs({nab_ij : Nspec[2,1]})#.diff(X[2,1])
```




$\displaystyle \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}$




```python
# Substituting the value of the function n_ij first, and THEN differentiating with respect to something in that substitution. (X_21)
sigma(nab_ij).subs({nab_ij : Nspec[2,1]}).diff(X[2,1])
```




$\displaystyle w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} }}$




```python
Selem[2,1].subs({Nelem[2,1] : Nspec[2,1]}).diff(X[2,1])



```




$\displaystyle w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} }}$




```python
# ### WAY 2:
n_11 = Function('n_11')(Nspec[0, 0]); n_11
```




$\displaystyle \operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)}$




```python
sigma(n_11)
```




$\displaystyle \sigma{\left(\operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} \right)}$




```python
assert Nspec[0,0] == n_11.args[0]

sigma(n_11).subs({n_11 : n_11.args[0]})
```




$\displaystyle \sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)}$




```python
sigma(n_11).diff(n_11) #.replace(n_ij, n_ij.args[0])
```




$\displaystyle \frac{\partial}{\partial \operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)}} \sigma{\left(\operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} \right)}$




```python
sigma(n_11).diff(n_11).subs({n_11 : n_11.args[0]}).subs({X[0,0]:77777})
```




$\displaystyle \left. \frac{d}{d \xi} \sigma{\left(\xi \right)} \right|_{\substack{ \xi=77777 w_{11} + w_{21} x_{12} + w_{31} x_{13} }}$




```python
sigma(n_11).diff(n_11).subs({n_11 : n_11.args[0]}).replace(n_11.args[0], 23) # same as subs in this case
```




$\displaystyle \left. \frac{d}{d \xi} \sigma{\left(\xi \right)} \right|_{\substack{ \xi=23 }}$




```python
sigma(n_11).diff(X[0,0])
```




$\displaystyle w_{11} \frac{\partial}{\partial \operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)}} \sigma{\left(\operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} \right)} \left. \frac{d}{d \xi_{1}} \operatorname{n_{11}}{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }}$




```python
id = Lambda(x, x)

sigma(n_11).diff(X[0,0]).subs({n_11 : id})
```




$\displaystyle w_{11} \left. \frac{d}{d \xi_{1}} \operatorname{n_{11}}{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }} \left. \frac{d}{d \xi} \sigma{\left(\xi \right)} \right|_{\substack{ \xi=\left( x \mapsto x \right) }}$




```python
# NOTE: so I don't think WAY 2 is correct because here it doesn't simplify the derivative d n11 / d eps11, since this should equal 1 because now n11 = eps11. Correct one is below (repeated from above)
sigma(n_11).diff(X[0,0]).subs({n_11 : Nspec[0,0]})
```




$\displaystyle w_{11} \left. \frac{d}{d \xi_{1}} \operatorname{n_{11}}{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }} \left. \frac{d}{d \xi} \sigma{\left(\xi \right)} \right|_{\substack{ \xi=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }}$




```python
# CORRECT WAY 1
sigma(n_11).subs({n_11 : Nspec[0,0]}).diff(X[0,0])
```




$\displaystyle w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }}$




```python
# CORRECT WAY 2

sigma(nab_ij).subs({nab_ij : Nspec[0,0]}).diff(X[0,0])
```




$\displaystyle w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }}$




```python
# CORRECT WAY 3
Selem[2,1].subs({Nelem[2,1] : Nspec[2,1]}).diff(X[2,1])
```




$\displaystyle w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} }}$




```python
sigma(n_11) # WAY 1: sigma argument is already hardcoded
```




$\displaystyle \sigma{\left(\operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} \right)}$




```python
sigma(nab_ij) # Way 2: sigma argument is function of matrixsymbol (better than 1)
```




$\displaystyle \sigma{\left(\operatorname{n_{ij}}{\left(X,W \right)} \right)}$




```python
Selem[2,1] # WAY 3: sigma argument is just symbol and we replace it as function with argument hardcoded only later. (better than 2)



```




$\displaystyle \sigma{\left(n_{32} \right)}$




```python
L
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)}$




```python
assert Selem == S.replace(n, vN).replace(sigmaApply, sigmaApply_)

Selem
```




$\displaystyle \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]$




```python
L.replace(n, vN).replace(sigmaApply, sigmaApply_)
```




$\displaystyle \lambda{\left(\left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] \right)}$




```python
#L.replace(n, vN).replace(sigmaApply, sigmaApply_).diff(Nelem[0,0])
```


```python
Lsum = L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)
Lsum
```




$\displaystyle \sigma{\left(n_{11} \right)} + \sigma{\left(n_{12} \right)} + \sigma{\left(n_{21} \right)} + \sigma{\left(n_{22} \right)} + \sigma{\left(n_{31} \right)} + \sigma{\left(n_{32} \right)}$




```python
Lsum.diff(Nelem)
```




$\displaystyle \left[\begin{matrix}\frac{d}{d n_{11}} \sigma{\left(n_{11} \right)} & \frac{d}{d n_{12}} \sigma{\left(n_{12} \right)}\\\frac{d}{d n_{21}} \sigma{\left(n_{21} \right)} & \frac{d}{d n_{22}} \sigma{\left(n_{22} \right)}\\\frac{d}{d n_{31}} \sigma{\left(n_{31} \right)} & \frac{d}{d n_{32}} \sigma{\left(n_{32} \right)}\end{matrix}\right]$




```python
Lsum.subs(elemToSpec)#.diff(X[2,1])
```




$\displaystyle \sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} + \sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} + \sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} + \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)} + \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)} + \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}$




```python
Lsum.subs(elemToSpec).diff(X)
```




$\displaystyle \left[\begin{matrix}w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} }}\\w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} }}\\w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} }}\end{matrix}\right]$




```python

specToElemD = {v : k for k, v in elemToSpecD.items()}

Lsum.subs(elemToSpecD).diff(X).subs(specToElemD)
```




$\displaystyle \left[\begin{matrix}w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{11} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{12} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{11} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{12} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{11} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{12} }}\\w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{21} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{22} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{21} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{22} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{21} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{22} }}\\w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{31} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{32} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{31} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{32} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{31} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{32} }}\end{matrix}\right]$


