```python
from sympy import Matrix, Symbol, derive_by_array, Lambda, Function, MatrixSymbol, Derivative
from sympy import var
from sympy.abc import x, i, j, a, b, c, d



```


```python
def myvar(letter: str, i: int, j: int) -> Symbol:
    letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
    return letter_ij


ns,ms,ps = 3,3,2

X = Matrix(ns, ms, lambda i,j : myvar('x', i, j)); X
```




$\displaystyle \left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right]$




```python
W = Matrix(ms, ps, lambda i,j : myvar('w', i, j)); W
```




$\displaystyle \left[\begin{matrix}w_{11} & w_{12}\\w_{21} & w_{22}\\w_{31} & w_{32}\end{matrix}\right]$




```python
#TODO how to make matrix symbols commutative?
# A = MatrixSymbol('X',ns,ms, is_commutative=True); Matrix(A)
A = MatrixSymbol('X',ns,ms); Matrix(A)
B = MatrixSymbol('W',ms,ps)






```


```python
v = lambda a,b: a*b

vL = Lambda((a,b), a*b)

n = Function('v') #, Lambda((a,b), a*b))

vN = lambda mat1, mat2: Matrix(mat1.shape[0], mat2.shape[1], lambda i, j: Symbol("n_{}{}".format(i+1, j+1))); vN


Nelem = vN(X, W)
Nelem
```




$\displaystyle \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right]$




```python
Nspec = v(X,W)
Nspec
```




$\displaystyle \left[\begin{matrix}w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} & w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\\w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} & w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\\w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} & w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\end{matrix}\right]$




```python
#N = v(X, W); N
N = n(A,B)
N








```




$\displaystyle v{\left(X,W \right)}$




```python

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
```




$\displaystyle \sigma_{apply}{\left(v{\left(X,W \right)} \right)}$




```python
sigmaApply_(Nelem)
```




$\displaystyle \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]$




```python
sigmaApply_2(Nelem)
```




$\displaystyle \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]$




```python
#sigmaApply_2(A*B).diff(Matrix(A))
```


```python
Sspec = S.subs({A:X, B:W}).replace(n, v).replace(sigmaApply, sigmaApply_)
Sspec
```




$\displaystyle \left[\begin{matrix}\sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} & \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\\sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} & \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\\sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} & \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right]$




```python
Selem = S.replace(n, vN).replace(sigmaApply, sigmaApply_)
Selem
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
elemToMatArgD = dict(itertools.chain(*[[(Nelem[i, j], Function("n_{}{}".format(i+1,j+1))(A,B) ) for j in range(2)] for i in range(3)]))

elemToMatArg = list(elemToMatArgD.items())

Matrix(elemToMatArg)
```




$\displaystyle \left[\begin{matrix}n_{11} & \operatorname{n_{11}}{\left(X,W \right)}\\n_{12} & \operatorname{n_{12}}{\left(X,W \right)}\\n_{21} & \operatorname{n_{21}}{\left(X,W \right)}\\n_{22} & \operatorname{n_{22}}{\left(X,W \right)}\\n_{31} & \operatorname{n_{31}}{\left(X,W \right)}\\n_{32} & \operatorname{n_{32}}{\left(X,W \right)}\end{matrix}\right]$




```python
matargToSpecD = dict(zip(elemToMatArgD.values(), elemToSpecD.values()))

matargToSpec = list(matargToSpecD.items())

Matrix(matargToSpec)
```




$\displaystyle \left[\begin{matrix}\operatorname{n_{11}}{\left(X,W \right)} & w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13}\\\operatorname{n_{12}}{\left(X,W \right)} & w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\\\operatorname{n_{21}}{\left(X,W \right)} & w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23}\\\operatorname{n_{22}}{\left(X,W \right)} & w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\\\operatorname{n_{31}}{\left(X,W \right)} & w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33}\\\operatorname{n_{32}}{\left(X,W \right)} & w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\end{matrix}\right]$




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
temp = st[0,0].diff(X[0,0]); temp

#nt[0,0]

#temp.replace(Function("n_11")(nt[0,0].args[0]), nt[0,0].args[0])

#temp.subs({nt[0,0] : nt[0,0].args[0]})


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

L = lambd(S); L
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)}$




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
Lsum.subs(elemToSpecD)#.diff(X[2,1])
```




$\displaystyle \sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} + \sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} + \sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} + \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)} + \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)} + \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}$




```python
Lsum.subs(elemToSpecD).diff(X)
```




$\displaystyle \left[\begin{matrix}w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} }}\\w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} }}\\w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} }}\end{matrix}\right]$




```python
# METHOD 1: direct matrix diff
#
### END RESULT ACHIEVED HERE (this is the end result and the most specific form of the result of the  matrix differentiation, when sigma is unknown)
specToElemD = {v:k for k,v in elemToSpecD.items()}

assert Lsum == L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)
Lsum.subs(elemToSpecD).diff(X).subs(specToElemD)


```




$\displaystyle \left[\begin{matrix}w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{11} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{12} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{11} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{12} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{11} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{12} }}\\w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{21} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{22} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{21} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{22} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{21} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{22} }}\\w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{31} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{32} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{31} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{32} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{31} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{32} }}\end{matrix}\right]$




```python
# METHOD 2: doing matrix symbol diff
#
#### NOW DOING THE MATRIX SYMBOL DIFF EXPRESSION (trying to achieve a form that shows the chain rule w.r.t to matrix symbol)
Selem
```




$\displaystyle \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]$




```python
L
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)}$




```python
#L.replace(A, A.T).diff(A) #ERROR: fatal python error ... why??
```


```python
#L.replace(n,v).diff(A).replace(sigmaApply, sigmaApply_) # ERROR
#L.replace(n,vN).subs(elemToSpecFuncD).replace(sigmaApply, sigmaApply_).diff(X) # why the zero matrix?
```


```python
L.replace(n,v).diff(A)
```




$\displaystyle \left. \frac{d}{d \xi_{1}} \sigma_{apply}{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=X W }} \frac{\partial}{\partial \sigma_{apply}{\left(X W \right)}} \lambda{\left(\sigma_{apply}{\left(X W \right)} \right)} \frac{\partial}{\partial X} X W$




```python
L.replace(n,vL).diff(A)
```




$\displaystyle \left. \frac{d}{d \xi_{1}} \sigma_{apply}{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=X W }} \frac{\partial}{\partial \sigma_{apply}{\left(X W \right)}} \lambda{\left(\sigma_{apply}{\left(X W \right)} \right)} \frac{\partial}{\partial X} X W$




```python

```


```python

```


```python

```


```python

```


```python
#L.replace(n,v).diff(A).replace(lambd,lambd_) ### ERROR sigma object is not iterable
#L.replace(n,vL).diff(A).replace(sigmaApply, sigmaApply_)### ERROR
#L.replace(n,v).diff(A).replace(sigmaApply, sigmaApply_) ### ERROR dummy object has no attribute applyfunc
```


```python
#L.replace(sigmaApply, sigmaApply_).diff(A) # ERROR
# L.replace(lambd, lambd_) # ERROR

#L.replace(n, v).replace(sigmaApply, sigmaApply_2)# shows matrix results, too specific, want the function composition notation as below but just applied to the function v(X,W) in abstract way
### METHOD 0: (prepare by substituting n --> v, then sigmaApply --> sigma)
L.replace(n, v).replace(sigmaApply, sigmaApply_)#.replace(lambd, lambd_)
```




$\displaystyle \lambda{\left({\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right) \right)}$




```python

# NOTE: the point here is that even replacing with a sympy Lambda doesn't give same result as above since above uses the V.applyfunc(sigma) within the Lambda.
L.replace(sigmaApply, Lambda(d, sigma(d)))
```




$\displaystyle \lambda{\left(\sigma{\left(v{\left(X,W \right)} \right)} \right)}$




```python

vSym = Symbol('v', applyfunc=True)
L.replace(n(A,B), vSym)
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(v \right)} \right)}$




```python
#L.replace(n(A,B), vSym).replace(sigmaApply, sigmaApply_)# ERROR because Symbol has no atttribute applyfunc (that is the one we want though so must use matrix symbol which for some reason works instead of just an ordinary symbol v
#V = MatrixSymbol()
# Takes in the symbols A and B matrices and returns the matrix symbol with the shape that is supposed to result after A*B
V = lambda matA, matB: MatrixSymbol('V', matA.shape[0], matB.shape[1])
V
V(A,B)#.shape
```




$\displaystyle V$




```python
from sympy import symbols
#V = MatrixSymbol('V', X.shape[0], W.shape[1])
i, j = symbols('i j')
M = MatrixSymbol('M', i, j)# abstract shape
sigmaApply_L = Lambda(M, M.applyfunc(sigma))
lambda_L = Lambda(M, sum(M))
```


```python
sigmaApply_L(A)
```




$\displaystyle {\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({X}\right)$




```python
# TODO: trying to figure out how to write L so that it is in terms of lambdas so get the form (d ---> sigma(d) COMPOAED ((X,W) -> V)) instead of (sigmaApply(v(X,W)))
Vs = MatrixSymbol("Vs", A.shape[0], B.shape[1])
VL = Lambda((A,B), MatrixSymbol('V', A.shape[0], B.shape[1]))
VL
```




$\displaystyle \left( \left( X, \  W\right) \mapsto V \right)$




```python
L.replace(n, VL)#.replace(sigmaApply, sigmaApply_L).subs({V:VL})
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(V \right)} \right)}$




```python
L.replace(n, VL).replace(sigmaApply, sigmaApply_)#.subs({VL(A,B) : n(A,B)}) ### ERROR
# This is v(X,W) in Lambda form:
VL
```




$\displaystyle \left( \left( X, \  W\right) \mapsto V \right)$




```python
VL(A,B)
#L.subs({n: V})
```




$\displaystyle V$




```python
L.replace(n(A,B), VL(A,B))#.replace(sigmaApply, sigmaApply_).subs({V(A,B) : n})
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(V \right)} \right)}$




```python
lambd(sigmaApply(VL))
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(\left( \left( X, \  W\right) \mapsto V \right) \right)} \right)}$




```python
lambd(sigmaApply(VL)).replace(VL, n(A,B))
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)}$




```python
lambd(sigmaApply(VL)).diff(A)
```




$\displaystyle \left. \frac{d}{d \xi_{1}} \sigma_{apply}{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=\left( \left( X, \  W\right) \mapsto V \right) }} \frac{d}{d \sigma_{apply}{\left(\left( \left( X, \  W\right) \mapsto V \right) \right)}} \lambda{\left(\sigma_{apply}{\left(\left( \left( X, \  W\right) \mapsto V \right) \right)} \right)} \frac{d}{d X} \left(\left( \left( X, \  W\right) \mapsto V \right)\right)$




```python
lambd(sigmaApply(VL)).diff(A).replace(VL, n(A,B))
```




$\displaystyle \left. \frac{d}{d \xi_{1}} \sigma_{apply}{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=v{\left(X,W \right)} }} \frac{\partial}{\partial \sigma_{apply}{\left(v{\left(X,W \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)} \frac{\partial}{\partial X} v{\left(X,W \right)}$




```python
lambd(sigmaApply(VL))#.replace(sigmaApply, sigmaApply_)#replace(V, n(A,B)).replace(sigmaApply, sigmaApply_)
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(\left( \left( X, \  W\right) \mapsto V \right) \right)} \right)}$




```python
# GOAL: want both sigma_apply to be in ---> form composed with the above x,w ---> V form
#lambd(sigmaApply(V)).replace(V, Vs).replace(sigmaApply, sigmaApply_).replace(Vs, V(A,B))### ERROR
lambd(sigmaApply(n(A,B))).replace(n(A,B), VL)
sigmaApply_(A)
sigmaApply_L(A)
```




$\displaystyle {\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({X}\right)$




```python
sigmaApply(Vs).replace(sigmaApply, sigmaApply_)
```




$\displaystyle {\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({Vs}\right)$




```python
sigmaApply(VL(A,B)).replace(sigmaApply, sigmaApply_)#.replace(V(A,B), V)#.subs({sigmaApply: sigmaApply_L})
```




$\displaystyle {\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({V}\right)$




```python
#sigmaApply(Vs).subs({Vs : V, sigmaApply: sigmaApply_L}) ### ERROR must be matrix instance
#sigmaApply(Vs).replace(sigmaApply , sigmaApply_L).subs({Vs : V})
#sigmaApply(V).replace(sigmaApply, sigmaApply_L)
```


```python

sa = Lambda((A,B), VL)
sa
```




$\displaystyle \left( \left( X, \  W\right) \mapsto \left( \left( X, \  W\right) \mapsto V \right) \right)$




```python
### ALTERNATE try of declaring a sigma-apply kind of function
#sas = Lambda((A,B), Vs.applyfunc(sigma))
```


```python
Lambda((A,B), sigma(VL))
```




$\displaystyle \left( \left( X, \  W\right) \mapsto \sigma{\left(\left( \left( X, \  W\right) \mapsto V \right) \right)} \right)$




```python
Lambda((A,B), sigma(VL)).diff(A) # nothing useful with this format, and weird-wrong since doesn't do chain rule wi.r. to sigma
```




$\displaystyle \frac{d}{d X} \left(\left( \left( X, \  W\right) \mapsto \sigma{\left(\left( \left( X, \  W\right) \mapsto V \right) \right)} \right)\right)$




```python
Lambda((A,B), sigma(VL(A,B)))
```




$\displaystyle \left( \left( X, \  W\right) \mapsto \sigma{\left(V \right)} \right)$




```python
sas = Lambda((A,B), VL(A,B).applyfunc(sigma))

sas
```




$\displaystyle \left( \left( X, \  W\right) \mapsto {\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({V}\right) \right)$




```python
# YAY this works now I can replace MATRIX SYMBOLS with ordinary sympy LAMBDAS (replace cano only replace same kind of thing / type)
sigma(Vs).subs(Vs, VL)
#
```




$\displaystyle \sigma{\left(\left( \left( X, \  W\right) \mapsto V \right) \right)}$




```python
sas(A,B)
```




$\displaystyle {\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({V}\right)$




```python
# A.applyfunc(sigma).subs(A, VL)# subs method doesn't work here with applyfunc
L
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)}$




```python
#sas(A,B).replace(V, V(A,B))
```


```python
sigmaApply_L
```




$\displaystyle \left( M \mapsto {\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({M}\right) \right)$




```python
sigmaApply_L(M)
```




$\displaystyle {\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({M}\right)$




```python
#sigmaApply_LFake = Lambda(M, M.applyfunc(sigma))
sigmaApply(M).replace(sigmaApply, sigmaApply_L)
```




$\displaystyle {\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({M}\right)$




```python
#sigmaApply(M).replace(sigmaApply, sigmaApply_).subs(M, n(A,B))
n = Function("v", applyfunc=True)
#sigmaApply_(Vs.subs(Vs, Lambda((A,B), n(A,B))))
from sympy import lambdify
#sigma(lambdify([A,B], n(A,B)))

#inner = Lambda((A,B), n(A,B)); inner

#sigmaApply_(n(A,B))
#sigmaApply(inner).replace(sigmaApply, Lambda(A, sigma(A)))
```


```python
#sigmaApply_L(M).subs(M, inner)
Lambda(d, sigma(d))
```




$\displaystyle \left( d \mapsto \sigma{\left(d \right)} \right)$




```python
### CLOSEST ever gotten to function composition (?) with sympy ....
#Lambda(d, sigma(inner))
```


```python
#Lambda(d, sigma(inner)).diff(A)
```


```python
#Lambda(d, sigma(inner)).replace(inner, vL(A,B)).diff(A)
```


```python

```


```python

```


```python
# sigmaApply_L(M).subs(M, VL)# new subs method fails here too
#sigmaApply_(M).subs(M, VL)
```


```python
sigmaApply_L(M).diff(M)
```




$\displaystyle {\left( d \mapsto \frac{d}{d d} \sigma{\left(d \right)} \right)}_{\circ}\left({M}\right)$




```python

```


```python

```


```python
sigma(VL)#.replace(V, V(A,B))
```




$\displaystyle \sigma{\left(\left( \left( X, \  W\right) \mapsto V \right) \right)}$




```python
sigma(VL).replace(VL, VL(A,B))
```




$\displaystyle \sigma{\left(V \right)}$




```python
#sigma(V).replace(V, VL)
```


```python

```


```python

```


```python
f = Function('f')
xtoxL = Lambda(a, a)
xtox = lambda a: a

f(x).subs({x : xtoxL})
```




$\displaystyle f{\left(\left( x \mapsto x \right) \right)}$




```python
f(x).subs(x, xtox)# works but below one with replace doesn't. When replacing arg with function uses SUBS without dictionary (instead of replace)
```




$\displaystyle f{\left(x \right)}$




```python
# f(x).replace(x, xtox)### ERROR xtox expects one positional argument ( I think replace only replaces the same kind of thing, never for instance a matrix symbol for a function or vice versa. the replacement needs to be of the same type / kind. But Lambda seems to work (as above))
f(x).replace(x, xtoxL)
```




$\displaystyle f{\left(\left( x \mapsto x \right) \right)}$




```python

```


```python

```


```python
#lambd(sigmaApply(n(A,B))).replace(n(A,B), Vs).replace(sigmaApply, sigmaApply_).replace(Vs, V)# ### ERROR rec replace must be matrix instance ....
```


```python

```


```python

```


```python

```


```python
### METHOD 0: the matrix diff rule in the most abstract form possible
n = Function("v", applyfunc=True) # necessary
L = lambd(sigmaApply(n(A,B)))

lambd_L = Lambda(A, sum(A))

lambd_L(A)
```




$\displaystyle X_{0, 0} + X_{0, 1} + X_{0, 2} + X_{1, 0} + X_{1, 1} + X_{1, 2} + X_{2, 0} + X_{2, 1} + X_{2, 2}$




```python
lambd_L(sigmaApply(n(A,B)))#.replace(n, vL).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_L)
```




$\displaystyle \left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)}\right)_{0, 0} + \left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)}\right)_{0, 1} + \left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)}\right)_{0, 2} + \left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)}\right)_{1, 0} + \left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)}\right)_{1, 1} + \left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)}\right)_{1, 2} + \left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)}\right)_{2, 0} + \left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)}\right)_{2, 1} + \left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)}\right)_{2, 2}$




```python
L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(A)
```




$\displaystyle \left. \frac{d}{d \xi_{1}} \lambda{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}={\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right) }} {\left( d \mapsto \frac{d}{d d} \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right) W^{T}$




```python
### SUCCESS! We see now that the matrix chain rule indeed makes the X transpose factor out on the left!!! (while compared to the above, the matrix transpose W^T factors out on the right, just like the book says (page 45 in the NOTE section of Seth Weidman book))
L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B)
```




$\displaystyle \left. \frac{d}{d \xi_{1}} \lambda{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}={\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right) }} X^{T} {\left( d \mapsto \frac{d}{d d} \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right)$




```python
# Not showing ???
L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B).replace(lambd, lambd_L)
```




$\displaystyle \left. \frac{d}{d \xi_{1}} \left(\xi_{1}_{0, 0} + \xi_{1}_{0, 1} + \xi_{1}_{0, 2} + \xi_{1}_{1, 0} + \xi_{1}_{1, 1} + \xi_{1}_{1, 2} + \xi_{1}_{2, 0} + \xi_{1}_{2, 1} + \xi_{1}_{2, 2}\right) \right|_{\substack{ \xi_{1}={\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right) }} X^{T} {\left( d \mapsto \frac{d}{d d} \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right)$




```python
#L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B).replace(B,W).replace(A,X) # ## ERROR non commutative scalars in matrix
# L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B).replace(lambd, lambd_).replace(B,W).replace(A,X)# ## ERROR dummy object not iterable
```


```python

#L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(A).replace(lambd, lambd_) ### ERROR: dummy object not iterable (probably means when in the above expression we have epsilon = sigmaApply(XW) that we cannot iterate over this expression)
```


```python
# Replacing lambda first: BAD
#L.replace(n, v).replace(lambd, lambd_) ## ERROR sigma apply object not ieterable
# Replacing sigma first: BAD
# L.replace(sigmaApply, sigmaApply_)### ERROR v object has no attribute applyfunc
```


```python
# Replacing n first: GOOD (need to go from inner nesting to outermost function, never any other way)
L.replace(n, v).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)
```




$\displaystyle \sigma{\left(W_{0, 0} X_{0, 0} + W_{1, 0} X_{0, 1} + W_{2, 0} X_{0, 2} \right)} + \sigma{\left(W_{0, 0} X_{1, 0} + W_{1, 0} X_{1, 1} + W_{2, 0} X_{1, 2} \right)} + \sigma{\left(W_{0, 0} X_{2, 0} + W_{1, 0} X_{2, 1} + W_{2, 0} X_{2, 2} \right)} + \sigma{\left(W_{0, 1} X_{0, 0} + W_{1, 1} X_{0, 1} + W_{2, 1} X_{0, 2} \right)} + \sigma{\left(W_{0, 1} X_{1, 0} + W_{1, 1} X_{1, 1} + W_{2, 1} X_{1, 2} \right)} + \sigma{\left(W_{0, 1} X_{2, 0} + W_{1, 1} X_{2, 1} + W_{2, 1} X_{2, 2} \right)}$




```python
# ### END RESULT of METHOD 2:
L.replace(n, v).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).diff(Matrix(A))
```




$\displaystyle \left[\begin{matrix}W_{0, 0} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 0} X_{0, 0} + W_{1, 0} X_{0, 1} + W_{2, 0} X_{0, 2} }} + W_{0, 1} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 1} X_{0, 0} + W_{1, 1} X_{0, 1} + W_{2, 1} X_{0, 2} }} & W_{1, 0} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 0} X_{0, 0} + W_{1, 0} X_{0, 1} + W_{2, 0} X_{0, 2} }} + W_{1, 1} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 1} X_{0, 0} + W_{1, 1} X_{0, 1} + W_{2, 1} X_{0, 2} }} & W_{2, 0} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 0} X_{0, 0} + W_{1, 0} X_{0, 1} + W_{2, 0} X_{0, 2} }} + W_{2, 1} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 1} X_{0, 0} + W_{1, 1} X_{0, 1} + W_{2, 1} X_{0, 2} }}\\W_{0, 0} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 0} X_{1, 0} + W_{1, 0} X_{1, 1} + W_{2, 0} X_{1, 2} }} + W_{0, 1} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 1} X_{1, 0} + W_{1, 1} X_{1, 1} + W_{2, 1} X_{1, 2} }} & W_{1, 0} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 0} X_{1, 0} + W_{1, 0} X_{1, 1} + W_{2, 0} X_{1, 2} }} + W_{1, 1} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 1} X_{1, 0} + W_{1, 1} X_{1, 1} + W_{2, 1} X_{1, 2} }} & W_{2, 0} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 0} X_{1, 0} + W_{1, 0} X_{1, 1} + W_{2, 0} X_{1, 2} }} + W_{2, 1} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 1} X_{1, 0} + W_{1, 1} X_{1, 1} + W_{2, 1} X_{1, 2} }}\\W_{0, 0} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 0} X_{2, 0} + W_{1, 0} X_{2, 1} + W_{2, 0} X_{2, 2} }} + W_{0, 1} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 1} X_{2, 0} + W_{1, 1} X_{2, 1} + W_{2, 1} X_{2, 2} }} & W_{1, 0} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 0} X_{2, 0} + W_{1, 0} X_{2, 1} + W_{2, 0} X_{2, 2} }} + W_{1, 1} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 1} X_{2, 0} + W_{1, 1} X_{2, 1} + W_{2, 1} X_{2, 2} }} & W_{2, 0} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 0} X_{2, 0} + W_{1, 0} X_{2, 1} + W_{2, 0} X_{2, 2} }} + W_{2, 1} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=W_{0, 1} X_{2, 0} + W_{1, 1} X_{2, 1} + W_{2, 1} X_{2, 2} }}\end{matrix}\right]$



Compare the above matrix symbol way with the Lsum way:

### END RESULT of METHOD 1:


```python
#Lsum = L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)

L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(X)#.subs(specToElemD)
```




$\displaystyle \left[\begin{matrix}w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} }}\\w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} }}\\w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} }}\end{matrix}\right]$




```python

```

COMPARING METHOD 0 (abstract way) with METHOD 2 (direct way) when differentiating .w.r.t to X vs. w.r.t to W
### With respect to X (abstract)


```python
L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(A)
```




$\displaystyle \left. \frac{d}{d \xi_{1}} \lambda{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}={\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right) }} {\left( d \mapsto \frac{d}{d d} \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right) W^{T}$



### With respect to X (direct)


```python
L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(X).subs(specToElemD)
```




$\displaystyle \left[\begin{matrix}w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{11} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{12} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{11} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{12} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{11} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{12} }}\\w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{21} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{22} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{21} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{22} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{21} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{22} }}\\w_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{31} }} + w_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{32} }} & w_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{31} }} + w_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{32} }} & w_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{31} }} + w_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{32} }}\end{matrix}\right]$



### With respect to W (abstract)


```python
L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(B)
```




$\displaystyle \left. \frac{d}{d \xi_{1}} \lambda{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}={\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right) }} X^{T} {\left( d \mapsto \frac{d}{d d} \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right)$



### With respect to W (direct)


```python
L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(W).subs(specToElemD)
```




$\displaystyle \left[\begin{matrix}x_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{11} }} + x_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{21} }} + x_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{31} }} & x_{11} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{12} }} + x_{21} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{22} }} + x_{31} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{32} }}\\x_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{11} }} + x_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{21} }} + x_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{31} }} & x_{12} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{12} }} + x_{22} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{22} }} + x_{32} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{32} }}\\x_{13} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{11} }} + x_{23} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{21} }} + x_{33} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{31} }} & x_{13} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{12} }} + x_{23} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{22} }} + x_{33} \left. \frac{d}{d \xi_{1}} \sigma{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=n_{32} }}\end{matrix}\right]$




```python

```


```python

```


```python

```


```python

```

### NEXT: try to substitute the X, W matrices step by step to see if you can come to the same result as the direct forms above (from method 2 or 1)


```python
from sympy import simplify, expand

#simplify(L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(B).subs({A:X, B:W})) ### ERROR max recursion depth exceeded
L.replace(n,vL).replace(sigmaApply, sigmaApply_).diff(B).subs({A:X, B:W})
```




$\displaystyle \left. \frac{d}{d \xi_{1}} \lambda{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}={\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({\left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right] \left[\begin{matrix}w_{11} & w_{12}\\w_{21} & w_{22}\\w_{31} & w_{32}\end{matrix}\right]}\right) }} \left(\left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right]\right)^{T} {\left( d \mapsto \frac{d}{d d} \sigma{\left(d \right)} \right)}_{\circ}\left({\left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right] \left[\begin{matrix}w_{11} & w_{12}\\w_{21} & w_{22}\\w_{31} & w_{32}\end{matrix}\right]}\right)$




```python
L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B)#.subs({A:X, B:W})
```




$\displaystyle \left. \frac{d}{d \xi_{1}} \lambda{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}={\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right) }} X^{T} {\left( d \mapsto \frac{d}{d d} \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right)$




```python
L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B)
```




$\displaystyle \left. \frac{d}{d \xi_{1}} \lambda{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}={\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right) }} X^{T} {\left( d \mapsto \frac{d}{d d} \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right)$




```python
#L.replace(n,v).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)
```


```python
#L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(B).subs({A:X, B:W}).replace(lambd, lambd_) ### ERROR dummy object not iterable
L.replace(n,v).diff(A)
```




$\displaystyle \left. \frac{d}{d \xi_{1}} \sigma_{apply}{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}=X W }} \frac{\partial}{\partial \sigma_{apply}{\left(X W \right)}} \lambda{\left(\sigma_{apply}{\left(X W \right)} \right)} \frac{\partial}{\partial X} X W$




```python

```


```python
#L.replace(n,v).replace(sigmaApply, sigmaApply_).diff(A).replace(A,Matrix(A))##ERROR noncommutative matrix scalars
# WANT: to be able to do diff and have the expression come out as above with X^T on left and W^T on right, when using just this form, with abstract form v:
L.replace(A,A.T).replace(B,B.T)
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(v{\left(X^{T},W^{T} \right)} \right)} \right)}$




```python
# Error if applying sigma to the v function because it sais v has no attribute applyfunc to trying now to making it have the attriute applyfunc.
y = Function('y', applyfunc=True, real=True)


```


```python
Ly = lambd(sigmaApply(y(A,B)))
Ly
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(y{\left(X,W \right)} \right)} \right)}$




```python

Ly.replace(A,A.T).replace(B,B.T)#.replace(sigmaApply, sigmaApply_)
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(y{\left(X^{T},W^{T} \right)} \right)} \right)}$




```python
# TODO next step: to apply the sigma to get that applied functor expression but here get error saying bol object not callable ...??
Ly.replace(A,A.T).replace(B,B.T)#.replace(sigmaApply, sigmaApply_)
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(y{\left(X^{T},W^{T} \right)} \right)} \right)}$




```python
# TODO always get fatal python error here, as if it can't deal with two matrix args!!
#Ly.replace(A,A.T).replace(B,B.T).diff(A)

#siga2 = Lambda(a, siga(a))
```


```python
Ly.replace(A, A.T).replace(B, b).diff(b)#.replace(sigmaApply, siga)
```




$\displaystyle \frac{\partial}{\partial \sigma_{apply}{\left(y{\left(X^{T},b \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(y{\left(X^{T},b \right)} \right)} \right)} \frac{\partial}{\partial y{\left(X^{T},b \right)}} \sigma_{apply}{\left(y{\left(X^{T},b \right)} \right)} \left(\frac{\partial}{\partial b} y{\left(X^{T},b \right)} + \mathbb{0}\right)$




```python
L.replace(n, vN).replace(sigmaApply, sigmaApply_).subs(elemToMatArgD)
```




$\displaystyle \lambda{\left(\left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(X,W \right)} \right)}\end{matrix}\right] \right)}$




```python
#L.replace(n, vN).replace(sigmaApply, sigmaApply_).subs(elemToMatArgD).diff(A)## ERROR: max recursion depth eceeded

L.replace(n, vN).replace(sigmaApply, sigmaApply_).subs(elemToMatArgD).diff(Matrix(3,2,list(elemToMatArgD.values())))
```




$\displaystyle \left[\begin{matrix}\left[\begin{matrix}\frac{\partial}{\partial \operatorname{n_{11}}{\left(X,W \right)}} \sigma{\left(\operatorname{n_{11}}{\left(X,W \right)} \right)} \frac{\partial}{\partial \left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(X,W \right)} \right)}\end{matrix}\right]} \lambda{\left(\left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(X,W \right)} \right)}\end{matrix}\right] \right)} & 0\\0 & 0\\0 & 0\end{matrix}\right] & \left[\begin{matrix}0 & \frac{\partial}{\partial \operatorname{n_{12}}{\left(X,W \right)}} \sigma{\left(\operatorname{n_{12}}{\left(X,W \right)} \right)} \frac{\partial}{\partial \left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(X,W \right)} \right)}\end{matrix}\right]} \lambda{\left(\left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(X,W \right)} \right)}\end{matrix}\right] \right)}\\0 & 0\\0 & 0\end{matrix}\right]\\\left[\begin{matrix}0 & 0\\\frac{\partial}{\partial \operatorname{n_{21}}{\left(X,W \right)}} \sigma{\left(\operatorname{n_{21}}{\left(X,W \right)} \right)} \frac{\partial}{\partial \left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(X,W \right)} \right)}\end{matrix}\right]} \lambda{\left(\left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(X,W \right)} \right)}\end{matrix}\right] \right)} & 0\\0 & 0\end{matrix}\right] & \left[\begin{matrix}0 & 0\\0 & \frac{\partial}{\partial \operatorname{n_{22}}{\left(X,W \right)}} \sigma{\left(\operatorname{n_{22}}{\left(X,W \right)} \right)} \frac{\partial}{\partial \left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(X,W \right)} \right)}\end{matrix}\right]} \lambda{\left(\left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(X,W \right)} \right)}\end{matrix}\right] \right)}\\0 & 0\end{matrix}\right]\\\left[\begin{matrix}0 & 0\\0 & 0\\\frac{\partial}{\partial \operatorname{n_{31}}{\left(X,W \right)}} \sigma{\left(\operatorname{n_{31}}{\left(X,W \right)} \right)} \frac{\partial}{\partial \left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(X,W \right)} \right)}\end{matrix}\right]} \lambda{\left(\left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(X,W \right)} \right)}\end{matrix}\right] \right)} & 0\end{matrix}\right] & \left[\begin{matrix}0 & 0\\0 & 0\\0 & \frac{\partial}{\partial \operatorname{n_{32}}{\left(X,W \right)}} \sigma{\left(\operatorname{n_{32}}{\left(X,W \right)} \right)} \frac{\partial}{\partial \left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(X,W \right)} \right)}\end{matrix}\right]} \lambda{\left(\left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(X,W \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(X,W \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(X,W \right)} \right)}\end{matrix}\right] \right)}\end{matrix}\right]\end{matrix}\right]$




```python
A.applyfunc(sigma)
```




$\displaystyle {\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({X}\right)$




```python
sigma = Function("sigma", applyfunc=True, bool=False)
```


```python
sigma.__dict__
```




    mappingproxy({'applyfunc': True,
                  'bool': False,
                  '_kwargs': {'applyfunc': True, 'bool': False},
                  '__module__': None,
                  '__doc__': None,
                  'name': 'sigma',
                  '_sage_': <sympy.core.function.UndefSageHelper at 0x7f52b8568e50>,
                  '_nargs': None,
                  '__sympy__': <property at 0x7f52a2dc6050>,
                  '_explicit_class_assumptions': {},
                  'default_assumptions': {},
                  '_prop_handler': {'negative': <function sympy.core.expr.Expr._eval_is_negative(self)>,
                   'positive': <function sympy.core.expr.Expr._eval_is_positive(self)>,
                   'commutative': <function sympy.core.function.Function._eval_is_commutative(self)>,
                   'extended_positive': <function sympy.core.expr.Expr._eval_is_extended_positive(self)>,
                   'extended_negative': <function sympy.core.expr.Expr._eval_is_extended_negative(self)>}})




```python
Ly = lambd(sigmaApply(y(A,B))); Ly
```




$\displaystyle \lambda{\left(\sigma_{apply}{\left(y{\left(X,W \right)} \right)} \right)}$




```python
(X*W).applyfunc(sigma)
```




$\displaystyle \left[\begin{matrix}\sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} & \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\\sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} & \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\\sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} & \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right]$




```python
(A*B).applyfunc(sigma)
```




$\displaystyle {\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right)$




```python
siga(A)
#A.applyfunc(siga) ### ERROR dumy object has no attribute shape
```




$\displaystyle \left[\begin{matrix}\sigma{\left(X_{0, 0} \right)} & \sigma{\left(X_{0, 1} \right)} & \sigma{\left(X_{0, 2} \right)}\\\sigma{\left(X_{1, 0} \right)} & \sigma{\left(X_{1, 1} \right)} & \sigma{\left(X_{1, 2} \right)}\\\sigma{\left(X_{2, 0} \right)} & \sigma{\left(X_{2, 1} \right)} & \sigma{\left(X_{2, 2} \right)}\end{matrix}\right]$




```python
y = Function("y", applyfunc = True, bool=False, shape=(3,3))
y.shape
```




    (3, 3)




```python
# siga(y(A,B))### ERROR: function y is not subscriptable
```


```python

```


```python

```


```python

```


```python

```


```python
Ly.subs({A:a,B:b}).diff(b).subs({a:A, b:B})#.replace(sigmaApply, sigmaApply_)
```




$\displaystyle \frac{\partial}{\partial \sigma_{apply}{\left(y{\left(X,W \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(y{\left(X,W \right)} \right)} \right)} \frac{\partial}{\partial y{\left(X,W \right)}} \sigma_{apply}{\left(y{\left(X,W \right)} \right)} \frac{\partial}{\partial W} y{\left(X,W \right)}$




```python
L.replace(A,a).replace(B,b).diff(b).subs({a:A,b:B})#.replace(sigmaApply, sigmaApply_)#.diff(b)
```




$\displaystyle \frac{\partial}{\partial \sigma_{apply}{\left(v{\left(X,W \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)} \frac{\partial}{\partial v{\left(X,W \right)}} \sigma_{apply}{\left(v{\left(X,W \right)} \right)} \frac{\partial}{\partial W} v{\left(X,W \right)}$




```python
sigma = Function("sigma", applyfunc=True, real=True)
sigmaApply_ = lambda mat: mat.applyfunc(sigma)
L = lambd(sigmaApply(n(A,B)))

#L.replace(A,a).replace(B,b).diff(b).subs({a:A,b:B}).replace(sigmaApply, sigmaApply_)
L.replace(n, v).replace(sigmaApply, sigmaApply_).diff(A)
#m = Symbol("m", shape=(3,2))
#m.shape

#sigmaApply_3 = Lambda(m, siga(m))

#L.replace(A,a).replace(B,b).diff(b).replace(b,B).replace(a,A).subs({n:vL}).replace(sigmaApply, sigmaApply_2) ### ERROR: Dummy object has no attribute shape
```




$\displaystyle \left. \frac{d}{d \xi_{1}} \lambda{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}={\left( d \mapsto \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right) }} {\left( d \mapsto \frac{d}{d d} \sigma{\left(d \right)} \right)}_{\circ}\left({X W}\right) W^{T}$




```python
# Ly.replace(B, b).diff(A)#.replace(sigmaApply, siga)### ERROR noncommutative matrix scalars not supported
Ly.replace(A, A.T).replace(B, b).diff(b).replace(b, B).replace(A.T, A)#.replace(sigmaApply, siga)
```




$\displaystyle \frac{\partial}{\partial \sigma_{apply}{\left(y{\left(X,W \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(y{\left(X,W \right)} \right)} \right)} \frac{\partial}{\partial y{\left(X,W \right)}} \sigma_{apply}{\left(y{\left(X,W \right)} \right)} \left(\frac{\partial}{\partial W} y{\left(X,W \right)} + \mathbb{0}\right)$




```python
#Ly.replace(B,b).diff(b).replace(b,B) ### ERROR
```


```python
# NEXT: try to replace the sigma apply, not working
n.__dict__
```




    mappingproxy({'applyfunc': True,
                  '_kwargs': {'applyfunc': True},
                  '__module__': None,
                  '__doc__': None,
                  'name': 'v',
                  '_sage_': <sympy.core.function.UndefSageHelper at 0x7f52b8568e50>,
                  '_nargs': None,
                  '__sympy__': <property at 0x7f52a2fefe90>,
                  '_explicit_class_assumptions': {},
                  'default_assumptions': {},
                  '_prop_handler': {'negative': <function sympy.core.expr.Expr._eval_is_negative(self)>,
                   'positive': <function sympy.core.expr.Expr._eval_is_positive(self)>,
                   'commutative': <function sympy.core.function.Function._eval_is_commutative(self)>,
                   'extended_positive': <function sympy.core.expr.Expr._eval_is_extended_positive(self)>,
                   'extended_negative': <function sympy.core.expr.Expr._eval_is_extended_negative(self)>}})




```python
y.__dict__
# TODO HERE
#https://stackoverflow.com/questions/12614334/typeerror-bool-object-is-not-callable
```




    mappingproxy({'applyfunc': True,
                  'bool': False,
                  'shape': (3, 3),
                  '_kwargs': {'applyfunc': True, 'bool': False, 'shape': (3, 3)},
                  '__module__': None,
                  '__doc__': None,
                  'name': 'y',
                  '_sage_': <sympy.core.function.UndefSageHelper at 0x7f52b8568e50>,
                  '_nargs': None,
                  '__sympy__': <property at 0x7f52a2e15530>,
                  '_explicit_class_assumptions': {},
                  'default_assumptions': {},
                  '_prop_handler': {'negative': <function sympy.core.expr.Expr._eval_is_negative(self)>,
                   'positive': <function sympy.core.expr.Expr._eval_is_positive(self)>,
                   'commutative': <function sympy.core.function.Function._eval_is_commutative(self)>,
                   'extended_positive': <function sympy.core.expr.Expr._eval_is_extended_positive(self)>,
                   'extended_negative': <function sympy.core.expr.Expr._eval_is_extended_negative(self)>}})




```python

```


```python

```


```python
from sympy import diff
# ### WARNING: this only works when size(X) == size(Y) else since size(W) != size(X) cannot subst B with W, so this operation won't work in my case.

#X = Matrix(3,3, lambda i,j: Symbol("x_{}{}".format(i+1,j+1))); Matrix(X)
# Create another matrix instead of W so that it matches size of X during diff(X) operation, since otherwise the diff by X doesn't work, says X and W need to be same size.

Wtemp = Matrix(*X.shape, lambda i,j: Symbol("t_{}{}".format(i+1,j+1))); Matrix(Wtemp)
```




$\displaystyle \left[\begin{matrix}t_{11} & t_{12} & t_{13}\\t_{21} & t_{22} & t_{23}\\t_{31} & t_{32} & t_{33}\end{matrix}\right]$




```python
#L.subs({A:X, B:Wtemp}).diff(X)[0,0][0,0].replace(n,vN).replace(sigmaApply, sigmaApply_)#.doit()
#diff(L.replace(A,A.T), A) # ERROR max recursion depth exceeded
```


```python
#Lmat = L.subs({A:X, B:Wtemp}).diff(X).subs({X:A, Wtemp: B}); Lmat #replace(X, A).replace(Y,B); Lmat
# NOTE need to do replace at the end (instead of subs) else it says unhasable type mutabledensematrix.
Lmat = L.subs({A:X, B:Wtemp}).diff(X).replace(X, A).replace(Wtemp,B); Lmat
#L.diff(A) # HELL ON THE EDITOR NEVER TRY THIS AGAIN
```




$\displaystyle \left[\begin{matrix}\left[\begin{matrix}\frac{\partial}{\partial X} v{\left(X,W \right)} \frac{\partial}{\partial v{\left(X,W \right)}} \sigma_{apply}{\left(v{\left(X,W \right)} \right)} \frac{\partial}{\partial \sigma_{apply}{\left(v{\left(X,W \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)} & 0 & 0\\0 & 0 & 0\\0 & 0 & 0\end{matrix}\right] & \left[\begin{matrix}0 & \frac{\partial}{\partial X} v{\left(X,W \right)} \frac{\partial}{\partial v{\left(X,W \right)}} \sigma_{apply}{\left(v{\left(X,W \right)} \right)} \frac{\partial}{\partial \sigma_{apply}{\left(v{\left(X,W \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)} & 0\\0 & 0 & 0\\0 & 0 & 0\end{matrix}\right] & \left[\begin{matrix}0 & 0 & \frac{\partial}{\partial X} v{\left(X,W \right)} \frac{\partial}{\partial v{\left(X,W \right)}} \sigma_{apply}{\left(v{\left(X,W \right)} \right)} \frac{\partial}{\partial \sigma_{apply}{\left(v{\left(X,W \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)}\\0 & 0 & 0\\0 & 0 & 0\end{matrix}\right]\\\left[\begin{matrix}0 & 0 & 0\\\frac{\partial}{\partial X} v{\left(X,W \right)} \frac{\partial}{\partial v{\left(X,W \right)}} \sigma_{apply}{\left(v{\left(X,W \right)} \right)} \frac{\partial}{\partial \sigma_{apply}{\left(v{\left(X,W \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)} & 0 & 0\\0 & 0 & 0\end{matrix}\right] & \left[\begin{matrix}0 & 0 & 0\\0 & \frac{\partial}{\partial X} v{\left(X,W \right)} \frac{\partial}{\partial v{\left(X,W \right)}} \sigma_{apply}{\left(v{\left(X,W \right)} \right)} \frac{\partial}{\partial \sigma_{apply}{\left(v{\left(X,W \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)} & 0\\0 & 0 & 0\end{matrix}\right] & \left[\begin{matrix}0 & 0 & 0\\0 & 0 & \frac{\partial}{\partial X} v{\left(X,W \right)} \frac{\partial}{\partial v{\left(X,W \right)}} \sigma_{apply}{\left(v{\left(X,W \right)} \right)} \frac{\partial}{\partial \sigma_{apply}{\left(v{\left(X,W \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)}\\0 & 0 & 0\end{matrix}\right]\\\left[\begin{matrix}0 & 0 & 0\\0 & 0 & 0\\\frac{\partial}{\partial X} v{\left(X,W \right)} \frac{\partial}{\partial v{\left(X,W \right)}} \sigma_{apply}{\left(v{\left(X,W \right)} \right)} \frac{\partial}{\partial \sigma_{apply}{\left(v{\left(X,W \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)} & 0 & 0\end{matrix}\right] & \left[\begin{matrix}0 & 0 & 0\\0 & 0 & 0\\0 & \frac{\partial}{\partial X} v{\left(X,W \right)} \frac{\partial}{\partial v{\left(X,W \right)}} \sigma_{apply}{\left(v{\left(X,W \right)} \right)} \frac{\partial}{\partial \sigma_{apply}{\left(v{\left(X,W \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)} & 0\end{matrix}\right] & \left[\begin{matrix}0 & 0 & 0\\0 & 0 & 0\\0 & 0 & \frac{\partial}{\partial X} v{\left(X,W \right)} \frac{\partial}{\partial v{\left(X,W \right)}} \sigma_{apply}{\left(v{\left(X,W \right)} \right)} \frac{\partial}{\partial \sigma_{apply}{\left(v{\left(X,W \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)}\end{matrix}\right]\end{matrix}\right]$




```python
#L.replace(A,X).replace(B,W)
```


```python
# Method 2 approach for comparison:
#L.replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD).diff(X)#.subs(specToElemD)
```


```python
elem = Lmat[0,0][0,0];elem
```




$\displaystyle \frac{\partial}{\partial X} v{\left(X,W \right)} \frac{\partial}{\partial v{\left(X,W \right)}} \sigma_{apply}{\left(v{\left(X,W \right)} \right)} \frac{\partial}{\partial \sigma_{apply}{\left(v{\left(X,W \right)} \right)}} \lambda{\left(\sigma_{apply}{\left(v{\left(X,W \right)} \right)} \right)}$




```python
#Lmat.replace(n, vL) # error can't calc deriv .w.r.t to x11*w11 +...
# Lmat.replace(n, v) # error can't calc deriv .w.r.t to x11*w11 +...
elem.subs(n, vL)
```




$\displaystyle \left. \frac{d}{d \xi_{0}} \sigma_{apply}{\left(\xi_{0} \right)} \right|_{\substack{ \xi_{0}=X W }} \frac{\partial}{\partial X} X W \frac{\partial}{\partial \sigma_{apply}{\left(X W \right)}} \lambda{\left(\sigma_{apply}{\left(X W \right)} \right)}$




```python
#elem.replace(n, v) # error cannot deriv wrt to X*W
```


```python
Selem
```




$\displaystyle \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]$




```python
# use replace n with vN instead of subs n with vL to get less specific output so it is easier to see since vL returns the xww*w11 +.... expressions
elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_)
```




$\displaystyle \frac{\partial}{\partial \left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right]} \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right]} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]} \lambda{\left(\left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] \right)}$




```python
# Making matrix symbols again
Ss = MatrixSymbol('S', 3,2) #n by p
Ns = MatrixSymbol('N', 3,2) #n by p



short = elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_).replace(X,A).replace(Nelem, Ns).replace(Selem ,Ss)
short
```




$\displaystyle \frac{d}{d X} N \frac{\partial}{\partial N} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]} \lambda{\left(\left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] \right)}$




```python
# Now going back to matrix form just to apply the last function LAMBDA
elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_)
```




$\displaystyle \frac{\partial}{\partial \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]} \left(\sigma{\left(n_{11} \right)} + \sigma{\left(n_{12} \right)} + \sigma{\left(n_{21} \right)} + \sigma{\left(n_{22} \right)} + \sigma{\left(n_{31} \right)} + \sigma{\left(n_{32} \right)}\right) \frac{\partial}{\partial \left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right]} \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right]} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]$




```python
# Making each of the n_ijs a function
#elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecD)

Matrix(elemToSpecFuncArgs)
```




$\displaystyle \left[\begin{matrix}n_{11} & \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\n_{12} & \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\n_{21} & \operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\n_{22} & \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\n_{31} & \operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\n_{32} & \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right]$




```python

long = elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).subs(elemToSpecFuncArgsD)
long
```




$\displaystyle \frac{\partial}{\partial \left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)}\end{matrix}\right]} \left(\sigma{\left(\operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} + \sigma{\left(\operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} + \sigma{\left(\operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} + \sigma{\left(\operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} + \sigma{\left(\operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} + \sigma{\left(\operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)}\right) \frac{\partial}{\partial \left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right]} \left[\begin{matrix}\operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}\operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right]} \left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)}\end{matrix}\right]$




```python
# short version again:
short
```




$\displaystyle \frac{d}{d X} N \frac{\partial}{\partial N} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]} \lambda{\left(\left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] \right)}$




```python
# long.doit() # error as base exp thing
```


```python
# Trying step by step replacement approach:
elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).replace(Nelem, Ns).replace(X,A)
```




$\displaystyle \frac{\partial}{\partial \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]} \left(\sigma{\left(n_{11} \right)} + \sigma{\left(n_{12} \right)} + \sigma{\left(n_{21} \right)} + \sigma{\left(n_{22} \right)} + \sigma{\left(n_{31} \right)} + \sigma{\left(n_{32} \right)}\right) \frac{d}{d X} N \frac{\partial}{\partial N} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]$




```python
# Seeing if replacing the order of replacing Ns matrix with Xs matrix makes a difference: ...
step = elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).replace(Nelem, Ns).replace(X,A).doit()
step
```




$\displaystyle \left[\begin{matrix}\frac{d}{d X} N \frac{\partial}{\partial N} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] & \frac{d}{d X} N \frac{\partial}{\partial N} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]\\\frac{d}{d X} N \frac{\partial}{\partial N} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] & \frac{d}{d X} N \frac{\partial}{\partial N} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]\\\frac{d}{d X} N \frac{\partial}{\partial N} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] & \frac{d}{d X} N \frac{\partial}{\partial N} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]\end{matrix}\right]$




```python
elem.subs({A:X, B:W}).replace(n, vN).replace(sigmaApply, sigmaApply_).replace(lambd, lambd_).replace(X,A).replace(Nelem, Ns).doit()
```




$\displaystyle \left[\begin{matrix}\frac{d}{d X} N \frac{\partial}{\partial N} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] & \frac{d}{d X} N \frac{\partial}{\partial N} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]\\\frac{d}{d X} N \frac{\partial}{\partial N} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] & \frac{d}{d X} N \frac{\partial}{\partial N} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]\\\frac{d}{d X} N \frac{\partial}{\partial N} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] & \frac{d}{d X} N \frac{\partial}{\partial N} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]\end{matrix}\right]$




```python
step.replace(Ns, Nelem)
```




$\displaystyle \left[\begin{matrix}\frac{\partial}{\partial X} \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right]} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] & \frac{\partial}{\partial X} \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right]} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]\\\frac{\partial}{\partial X} \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right]} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] & \frac{\partial}{\partial X} \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right]} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]\\\frac{\partial}{\partial X} \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right]} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right] & \frac{\partial}{\partial X} \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right]} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]\end{matrix}\right]$




```python
#step.replace(Ns, Nelem).replace(A,X).doit()#error immutable dense array has no attribute as base exp ...
elem2 = step[0,0].replace(Ns, Nelem)
elem2.replace(A,X).subs(elemToSpecFuncArgsD)
```




$\displaystyle \frac{\partial}{\partial \left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right]} \left[\begin{matrix}\operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}\operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right]} \left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)}\end{matrix}\right]$




```python
#elem2.replace(A,X).subs(elemToSpecFuncArgsD).doit()
F = Nelem.subs(elemToSpecFuncArgsD); F
```




$\displaystyle \left[\begin{matrix}\operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right]$




```python
F[0,0].diff(X[0,0])
```




$\displaystyle \frac{\partial}{\partial x_{11}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}$




```python
F[0,0].diff(X)
```




$\displaystyle \left[\begin{matrix}\frac{\partial}{\partial x_{11}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{12}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{13}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{21}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{22}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{23}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{31}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{32}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{33}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right]$




```python
F.diff(X)
```




$\displaystyle \left[\begin{matrix}\left[\begin{matrix}\frac{\partial}{\partial x_{11}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{11}} \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{11}} \operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{11}} \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{11}} \operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{11}} \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right] & \left[\begin{matrix}\frac{\partial}{\partial x_{12}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{12}} \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{12}} \operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{12}} \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{12}} \operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{12}} \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right] & \left[\begin{matrix}\frac{\partial}{\partial x_{13}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{13}} \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{13}} \operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{13}} \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{13}} \operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{13}} \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right]\\\left[\begin{matrix}\frac{\partial}{\partial x_{21}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{21}} \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{21}} \operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{21}} \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{21}} \operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{21}} \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right] & \left[\begin{matrix}\frac{\partial}{\partial x_{22}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{22}} \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{22}} \operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{22}} \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{22}} \operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{22}} \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right] & \left[\begin{matrix}\frac{\partial}{\partial x_{23}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{23}} \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{23}} \operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{23}} \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{23}} \operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{23}} \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right]\\\left[\begin{matrix}\frac{\partial}{\partial x_{31}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{31}} \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{31}} \operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{31}} \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{31}} \operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{31}} \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right] & \left[\begin{matrix}\frac{\partial}{\partial x_{32}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{32}} \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{32}} \operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{32}} \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{32}} \operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{32}} \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right] & \left[\begin{matrix}\frac{\partial}{\partial x_{33}} \operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{33}} \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{33}} \operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{33}} \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\frac{\partial}{\partial x_{33}} \operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \frac{\partial}{\partial x_{33}} \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right]\end{matrix}\right]$




```python
argsToSpecD = dict(zip(elemToSpecFuncArgsD.values(), elemToSpecD.values()))
argsToSpec = list(argsToSpecD.items())
Matrix(argsToSpec)
```




$\displaystyle \left[\begin{matrix}\operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13}\\\operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\\\operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23}\\\operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\\\operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33}\\\operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\end{matrix}\right]$




```python
F[0,0].diff(X[0,0]).subs(argsToSpecD)#.subs({elemToSpecFuncArgs[0][1] : Nspec[0,0]})
```




$\displaystyle \frac{\partial}{\partial x_{11}} \left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13}\right)$




```python
F[0,0].diff(X[0,0]).subs(argsToSpecD).doit()
```




$\displaystyle w_{11}$




```python
# NOTE: using diff did not work, said immutable dense array cannot be subs-ed
derive_by_array(F, X).subs(argsToSpecD)
```




$\displaystyle \left[\begin{matrix}\left[\begin{matrix}\frac{\partial}{\partial x_{11}} \left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13}\right) & \frac{\partial}{\partial x_{11}} \left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\right)\\\frac{\partial}{\partial x_{11}} \left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23}\right) & \frac{\partial}{\partial x_{11}} \left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\right)\\\frac{\partial}{\partial x_{11}} \left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33}\right) & \frac{\partial}{\partial x_{11}} \left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\right)\end{matrix}\right] & \left[\begin{matrix}\frac{\partial}{\partial x_{12}} \left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13}\right) & \frac{\partial}{\partial x_{12}} \left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\right)\\\frac{\partial}{\partial x_{12}} \left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23}\right) & \frac{\partial}{\partial x_{12}} \left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\right)\\\frac{\partial}{\partial x_{12}} \left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33}\right) & \frac{\partial}{\partial x_{12}} \left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\right)\end{matrix}\right] & \left[\begin{matrix}\frac{\partial}{\partial x_{13}} \left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13}\right) & \frac{\partial}{\partial x_{13}} \left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\right)\\\frac{\partial}{\partial x_{13}} \left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23}\right) & \frac{\partial}{\partial x_{13}} \left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\right)\\\frac{\partial}{\partial x_{13}} \left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33}\right) & \frac{\partial}{\partial x_{13}} \left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\right)\end{matrix}\right]\\\left[\begin{matrix}\frac{\partial}{\partial x_{21}} \left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13}\right) & \frac{\partial}{\partial x_{21}} \left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\right)\\\frac{\partial}{\partial x_{21}} \left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23}\right) & \frac{\partial}{\partial x_{21}} \left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\right)\\\frac{\partial}{\partial x_{21}} \left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33}\right) & \frac{\partial}{\partial x_{21}} \left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\right)\end{matrix}\right] & \left[\begin{matrix}\frac{\partial}{\partial x_{22}} \left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13}\right) & \frac{\partial}{\partial x_{22}} \left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\right)\\\frac{\partial}{\partial x_{22}} \left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23}\right) & \frac{\partial}{\partial x_{22}} \left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\right)\\\frac{\partial}{\partial x_{22}} \left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33}\right) & \frac{\partial}{\partial x_{22}} \left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\right)\end{matrix}\right] & \left[\begin{matrix}\frac{\partial}{\partial x_{23}} \left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13}\right) & \frac{\partial}{\partial x_{23}} \left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\right)\\\frac{\partial}{\partial x_{23}} \left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23}\right) & \frac{\partial}{\partial x_{23}} \left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\right)\\\frac{\partial}{\partial x_{23}} \left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33}\right) & \frac{\partial}{\partial x_{23}} \left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\right)\end{matrix}\right]\\\left[\begin{matrix}\frac{\partial}{\partial x_{31}} \left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13}\right) & \frac{\partial}{\partial x_{31}} \left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\right)\\\frac{\partial}{\partial x_{31}} \left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23}\right) & \frac{\partial}{\partial x_{31}} \left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\right)\\\frac{\partial}{\partial x_{31}} \left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33}\right) & \frac{\partial}{\partial x_{31}} \left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\right)\end{matrix}\right] & \left[\begin{matrix}\frac{\partial}{\partial x_{32}} \left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13}\right) & \frac{\partial}{\partial x_{32}} \left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\right)\\\frac{\partial}{\partial x_{32}} \left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23}\right) & \frac{\partial}{\partial x_{32}} \left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\right)\\\frac{\partial}{\partial x_{32}} \left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33}\right) & \frac{\partial}{\partial x_{32}} \left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\right)\end{matrix}\right] & \left[\begin{matrix}\frac{\partial}{\partial x_{33}} \left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13}\right) & \frac{\partial}{\partial x_{33}} \left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\right)\\\frac{\partial}{\partial x_{33}} \left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23}\right) & \frac{\partial}{\partial x_{33}} \left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\right)\\\frac{\partial}{\partial x_{33}} \left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33}\right) & \frac{\partial}{\partial x_{33}} \left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\right)\end{matrix}\right]\end{matrix}\right]$




```python
derive_by_array(F, X).subs(argsToSpecD).doit()
```




$\displaystyle \left[\begin{matrix}\left[\begin{matrix}w_{11} & w_{12}\\0 & 0\\0 & 0\end{matrix}\right] & \left[\begin{matrix}w_{21} & w_{22}\\0 & 0\\0 & 0\end{matrix}\right] & \left[\begin{matrix}w_{31} & w_{32}\\0 & 0\\0 & 0\end{matrix}\right]\\\left[\begin{matrix}0 & 0\\w_{11} & w_{12}\\0 & 0\end{matrix}\right] & \left[\begin{matrix}0 & 0\\w_{21} & w_{22}\\0 & 0\end{matrix}\right] & \left[\begin{matrix}0 & 0\\w_{31} & w_{32}\\0 & 0\end{matrix}\right]\\\left[\begin{matrix}0 & 0\\0 & 0\\w_{11} & w_{12}\end{matrix}\right] & \left[\begin{matrix}0 & 0\\0 & 0\\w_{21} & w_{22}\end{matrix}\right] & \left[\begin{matrix}0 & 0\\0 & 0\\w_{31} & w_{32}\end{matrix}\right]\end{matrix}\right]$




```python
derive_by_array(F, W).subs(argsToSpecD).doit()



```




$\displaystyle \left[\begin{matrix}\left[\begin{matrix}x_{11} & 0\\x_{21} & 0\\x_{31} & 0\end{matrix}\right] & \left[\begin{matrix}0 & x_{11}\\0 & x_{21}\\0 & x_{31}\end{matrix}\right]\\\left[\begin{matrix}x_{12} & 0\\x_{22} & 0\\x_{32} & 0\end{matrix}\right] & \left[\begin{matrix}0 & x_{12}\\0 & x_{22}\\0 & x_{32}\end{matrix}\right]\\\left[\begin{matrix}x_{13} & 0\\x_{23} & 0\\x_{33} & 0\end{matrix}\right] & \left[\begin{matrix}0 & x_{13}\\0 & x_{23}\\0 & x_{33}\end{matrix}\right]\end{matrix}\right]$




```python
elem2
```




$\displaystyle \frac{\partial}{\partial X} \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right]} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]$




```python

funcMat = elem2.subs(elemToSpecFuncArgsD).replace(A,X)#.diff(X)
funcMat
```




$\displaystyle \frac{\partial}{\partial \left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right]} \left[\begin{matrix}\operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}\operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\\\operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} & \operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)}\end{matrix}\right]} \left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(x_{11},x_{12},x_{13},x_{21},x_{22},x_{23},x_{31},x_{32},x_{33},w_{11},w_{12},w_{21},w_{22},w_{31},w_{32} \right)} \right)}\end{matrix}\right]$




```python
#funcMat.doit() # error
#derive_by_array(funcMat, X)
```


```python
funcMat = elem2.subs(elemToSpecFuncD).replace(A,X)#.diff(X)
funcMat
```




$\displaystyle \frac{\partial}{\partial \left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right]} \left[\begin{matrix}\operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} & \operatorname{n_{12}}{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\\operatorname{n_{21}}{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} & \operatorname{n_{22}}{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\\operatorname{n_{31}}{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} & \operatorname{n_{32}}{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}\operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} & \operatorname{n_{12}}{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\\operatorname{n_{21}}{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} & \operatorname{n_{22}}{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\\operatorname{n_{31}}{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} & \operatorname{n_{32}}{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right]} \left[\begin{matrix}\sigma{\left(\operatorname{n_{11}}{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} \right)} & \sigma{\left(\operatorname{n_{12}}{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)} \right)}\\\sigma{\left(\operatorname{n_{21}}{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} \right)} & \sigma{\left(\operatorname{n_{22}}{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)} \right)}\\\sigma{\left(\operatorname{n_{31}}{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} \right)} & \sigma{\left(\operatorname{n_{32}}{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)} \right)}\end{matrix}\right]$




```python
#funcMat.doit() # same error
#elem2.subs(elemToSpecFuncD).doit() # error
elem2
```




$\displaystyle \frac{\partial}{\partial X} \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right] \frac{\partial}{\partial \left[\begin{matrix}n_{11} & n_{12}\\n_{21} & n_{22}\\n_{31} & n_{32}\end{matrix}\right]} \left[\begin{matrix}\sigma{\left(n_{11} \right)} & \sigma{\left(n_{12} \right)}\\\sigma{\left(n_{21} \right)} & \sigma{\left(n_{22} \right)}\\\sigma{\left(n_{31} \right)} & \sigma{\left(n_{32} \right)}\end{matrix}\right]$




```python
# elem2.replace(A,X).doit() # error
```


```python
#elem2.replace(A,a).doit()#.subs(elemToSpecFuncArgsD).doit()
# ERROR everywhere what next todo? this approach worked before, where I make w.r.t. thing a real matrix, and leave the others a symbol so why isn't it working now?
```


```python
#elem2.replace(A,X).subs(elemToSpecFuncD).doit()
# ERROR this has to work though! Then can simply replace n_ijs with lambda


```


```python
#elem2.subs(elemToMatArgD).doit()#ERROR max recursion depth exceeeded
```
