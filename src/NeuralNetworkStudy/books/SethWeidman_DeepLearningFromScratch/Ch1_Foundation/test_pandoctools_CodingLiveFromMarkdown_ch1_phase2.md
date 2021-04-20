---
pandoctools:
  profile: Default
  out: "*.ipynb"
  # out: "*.pdf"
input: True
eval: True
echo: True
error: raise
---



```py {input=False, echo=True, eval=True}
from sympy import Matrix, Symbol, derive_by_array, Lambda, Function, MatrixSymbol, Derivative, diff, symbols
from sympy import var
from sympy.abc import x, i, j, a, b


from sympy.interactive import init_printing

init_printing(pretty_print=True, wrap_line=True, num_columns=60)
```



```py {echo=True, input=False, eval=True}
def myvar(letter: str, i: int, j: int) -> Symbol:
    letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
    return letter_ij


n,m,p = 3,3,2

X = Matrix(n, m, lambda i,j : myvar('x', i, j)); X


from IPython.display import Markdown
from sympy import *
#latex(X)



Markdown(latex(X))
```

@{input=False, eval=True}
```py
W = Matrix(m, p, lambda i,j : myvar('w', i, j)); W
```

@{input=False, eval=True}
```py
A = MatrixSymbol('X',3,3); Matrix(A)
```
@{input=False, eval=True}
```py
B = MatrixSymbol('W',3,2)
```

# Defining N

Defining $N = \nu(X, W) = X \times W$

* $\nu : \mathbb{R}^{(n \times m) \times (m \times p)} \rightarrow \mathbb{R}^{n \times p}$
* $N \in \mathbb{R}^{n \times p}$

```py
v = lambda a,b: a*b

vL = Lambda((a,b), a*b)

n = Function('v') #, Lambda((a,b), a*b))

vN = lambda mat1, mat2: Matrix(mat1.shape[0], mat2.shape[1], lambda i, j: Symbol("n_{}{}".format(i+1, j+1))); vN

Nelem = vN(X, W); Nelem
```
