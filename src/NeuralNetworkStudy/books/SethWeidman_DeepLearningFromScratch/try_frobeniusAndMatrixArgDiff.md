```python title="codecell"
# SOURCE = https://www.kannon.link/free/2019/10/30/symbolic-matrix-differentiation-with-sympy/
from sympy import diff, symbols, MatrixSymbol, Transpose, Trace, Matrix, Function


def squared_frobenius_norm(expr):
    return Trace(expr * Transpose(expr))

k, m, n = symbols('k m n')

X = MatrixSymbol('X', m, k)
W = MatrixSymbol('W', k, n)
Y = MatrixSymbol('Y', m, n)

# Matrix(X)
A = MatrixSymbol('A', 3, 4)
B = MatrixSymbol('B', 4, 2)
C = MatrixSymbol('C', 3, 2)
Matrix(A)
```

```python title="codecell"
diff(squared_frobenius_norm(X*W - Y), W)
```
```python title="codecell"
sq = squared_frobenius_norm(A*B - C); sq
```
```python title="codecell"
diff(squared_frobenius_norm(A*B - C), B)
```

```python title="codecell"
sq.args[0]
```
```python title="codecell"
from sympy import srepr, expand, simplify, collect, factor, cancel, apart

#srepr(sq.args[0])
expand(sq.args[0])
```
```python title="codecell"
#diff(sq.args[0], B)
#diff(expand(sq.args[0]), B).doit()
from sympy import Symbol
Xm = Matrix(3,3, lambda i,j : Symbol("x_{}{}".format(i+1,j+1), commutative=True))
Wm = Matrix(3,2, lambda i,j : Symbol("w_{}{}".format(i+1,j+1), commutative=True))

X = MatrixSymbol('X',3,3)
W = MatrixSymbol('W', 3,2);
```

```python title="codecell"
diff(X*W, X)
```
```python title="codecell"
diff(X*W, X).subs({X:Xm})
```
```python title="codecell"
diff(X*W, X).subs({X:Xm}).doit()
```
```python title="codecell"
diff(X*W, X).subs({X:Xm}).doit().subs({W:Wm})
```

```python title="codecell"
# expand(diff(X*W, X).subs({X:Xm}).doit().subs({W:Wm}))# STUCK doesn't work to expand out from here
#diff(X*W, X).replace(X,Xm)# ERROR so I must use subs instead (noncommutatitve scalars in matrix multiplication not supported)
diff(X*W, X).subs({X:Xm, W:Wm}).doit()
```

```python title="codecell"
g,f = symbols('g f', cls = Function)
f(X).replace(X, X.T).diff(X).replace(X.T, X)
```
```python title="codecell"
g(f(X)).replace(X, X.T).diff(X).replace(X.T, X)
```
```python title="codecell"
# f(X,W).replace(X,X.T).diff(X)### CRASHES
```

```python title="codecell"

```
```python title="codecell"

```
```python title="codecell"

```
```python title="codecell"

```
```python title="codecell"



```

```python title="codecell"
type(sq.args[0])
```
```python title="codecell"
from sympy import symbols, Function

#h,g,f = symbols('h g f', cls=Function)
f = Function('f')
g = Function('g')
h = g(f(sq.args[0]))
h
```
```python title="codecell"
diff(h, B)
```

```python title="codecell"
from sympy import Derivative

#h.replace(f, Trace)
```

```python title="codecell"
diff(sq.args[0], B)
```
```python title="codecell"
from sympy import Trace


h = f(Trace(sq.args[0]))

diff(h, B)
```
```python title="codecell"
h = g(f(A*B))
h
```

```python title="codecell"
diff(h, A)
```
```python title="codecell"
from sympy import ZeroMatrix
Z = ZeroMatrix(3,4); Z
Matrix(Z)
```
```python title="codecell"
type(A.T)
```
```python title="codecell"
type(Z + A)
```
```python title="codecell"
type(A*1)
```
```python title="codecell"
type(A)
```
```python title="codecell"
type(A*B)
```
```python title="codecell"
from sympy.matrices.expressions.matexpr import MatrixExpr

#Matrix(MatrixExpr(A)) # ERROR
```
```python title="codecell"

```
```python title="codecell"
# diff(h, A) # WHAT THIS IS STILL BAD

# This is why:
assert type(A.T) != type(A.T.T)
#h = g(f(Z + A))
#D = MatrixSymbol('D', 3,4)

#ad = A+D
from sympy.abc import i,j,x,a,b,c

h = g(f(A.T))

h
```
```python title="codecell"

diff(h, A).replace(A.T,A)
```
```python title="codecell"
diff(A.T, A).replace(A.T, A)
```

```python title="codecell"
diff(A.T, A).replace(A, Matrix(A))#.doit()
```
```python title="codecell"
diff(A.T, A).replace(A, Matrix(A)).doit()
```


```python title="codecell"
from sympy import Symbol
from sympy.abc import b

#A = MatrixSymbol('A', 3,4)
M = Matrix(3,4, lambda i,j : Symbol('x_{}{}'.format(i+1,j+1)))
Matrix(M)
```
```python title="codecell"
Matrix(A)
```
```python title="codecell"
g, f = symbols('g f', cls = Function)

#__ = lambda mat: mat.T # transposes matrix symbol

diff( g(f(M,b)), b)
```

```python title="codecell"
diff( g(f(M,b)), b).replace(M, A)
```
```python title="codecell"
Ms = MatrixSymbol('M',2,2)
Ds = MatrixSymbol('D',2,2)
M = Matrix(2,2, lambda i,j: Symbol("m_{}{}".format(i+1,j+1)))
D = Matrix(2,2, lambda i,j: Symbol("d_{}{}".format(i+1,j+1)))

diff( g(f(M, D)), D )
```
```python title="codecell"
diff( g(f(M, D)), D ).replace(D, Ds).replace(M, Ms)
```
```python title="codecell"
diff(Ds,Ds).replace(Ds,D).doit()
```

```python title="codecell"
#diff( g(f(Ms, Ds.T)), Ds )#.replace(Ds.T, Ds)
```
```python title="codecell"

```
```python title="codecell"

```
```python title="codecell"

```
