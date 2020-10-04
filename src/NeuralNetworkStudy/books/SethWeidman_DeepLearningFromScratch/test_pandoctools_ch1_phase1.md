---
ccsDelim: ', '
ccsLabelSep: ' --- '
ccsTemplate: $$i$$$$ccsLabelSep$$$$t$$
chapDelim: .
chaptersDepth: 1
comments-map:
  js:
  - //
  - '/\*'
  - '\*/'
  py:
  - '\#'
  - ''''''''
  - ''''''''
  - '\"\"\"'
  - '\"\"\"'
  r:
  - '\#'
  - '\"'
  - '\"'
crossrefYaml: 'pandoc-crossref.yaml'
echo: true
eqLabels: arabic
eqnPrefix:
- eq.
- eqns.
eqnPrefixTemplate: $$p$$ $$i$$
error: raise
eval: true
figLabels: arabic
figPrefix:
- fig.
- figs.
figPrefixTemplate: $$p$$ $$i$$
figureTemplate: $$figureTitle$$ $$i$$$$titleDelim$$ $$t$$
figureTitle: Figure
kernels-map:
  py: python3
  r: ir
lastDelim: ', '
listingTemplate: $$listingTitle$$ $$i$$$$titleDelim$$ $$t$$
listingTitle: Listing
lofTitle: |
  List of Figures
  ===============
lolTitle: |
  List of Listings
  ================
lotTitle: |
  List of Tables
  ==============
lstLabels: arabic
lstPrefix:
- lst.
- lsts.
lstPrefixTemplate: $$p$$ $$i$$
pairDelim: ', '
pandoctools:
  out: '\*.md'
  profile: Kiwi
rangeDelim: '-'
refDelim: ', '
refIndexTemplate: $$i$$$$suf$$
secHeaderDelim: 
secHeaderTemplate: $$i$$$$secHeaderDelim$$$$t$$
secLabels: arabic
secPrefix:
- sec.
- secs.
secPrefixTemplate: $$p$$ $$i$$
sectionsDepth: 0
styles-map:
  py: python
subfigLabels: alpha a
subfigureChildTemplate: $$i$$
subfigureRefIndexTemplate: $$i$$$$suf$$ ($$s$$)
subfigureTemplate: $$figureTitle$$ $$i$$$$titleDelim$$ $$t$$. $$ccs$$
tableTemplate: $$tableTitle$$ $$i$$$$titleDelim$$ $$t$$
tableTitle: Table
tblLabels: arabic
tblPrefix:
- tbl.
- tbls.
tblPrefixTemplate: $$p$$ $$i$$
titleDelim: ':'
---

``` {.python}
from IPython.display import Markdown

from sympy import Matrix, Symbol, derive_by_array, Lambda, symbols, Derivative, diff
from sympy.abc import x, y, i, j, a, b
```

``` {.python}
# Defining variable-element matrices $X \in \mathbb{R}^{n \times m}$ and $W \in \mathbb{R}^{m \times p}$:
```

``` {.python}
def var(letter: str, i: int, j: int) -> Symbol:
    letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
    return letter_ij


n,m,p = 3,3,2

X = Matrix(n, m, lambda i,j : var('x', i, j)); X
```

```{=latex}
$\displaystyle \left[\begin{matrix}x_{11} & x_{12} & x_{13}\\x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}\end{matrix}\right]$
```
``` {.python}
W = Matrix(m, p, lambda i,j : var('w', i, j)); W
```

```{=latex}
$\displaystyle \left[\begin{matrix}w_{11} & w_{12}\\w_{21} & w_{22}\\w_{31} & w_{32}\end{matrix}\right]$
```
``` {.python}
# Defining $N = \nu(X, W) = X \times W$
#
# * $\nu : \mathbb{R}^{(n \times m) \times (m \times p)} \rightarrow \mathbb{R}^{n \times p}$
# * $N \in \mathbb{R}^{n \times p}$
```

``` {.python}
v = Lambda((a,b), a*b); v
```

```{=latex}
$\displaystyle \left( \left( a, \  b\right) \mapsto a b \right)$
```
``` {.python}
N = v(X, W); N
```

```{=latex}
$\displaystyle \left[\begin{matrix}w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} & w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13}\\w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} & w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23}\\w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} & w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33}\end{matrix}\right]$
```
``` {.python}
# Defining $S = \sigma_{\text{apply}}(N) = \sigma_{\text{apply}}(\nu(X,W)) = \sigma_\text{apply}(X \times W) = \Big \{ \sigma(XW_{ij}) \Big\}$.
#
#
# Assume that $\sigma_{\text{apply}} : \mathbb{R}^{n \times p} \rightarrow \mathbb{R}^{n \times p}$ while $\sigma : \mathbb{R} \rightarrow \mathbb{R}$, so the function $\sigma_{\text{apply}}$ takes in a matrix and returns a matrix while the simple $\sigma$ acts on the individual elements $N_{ij} = XW_{ij}$ in the matrix argument $N$ of $\sigma_{\text{apply}}$.
#
# * $\sigma : \mathbb{R} \rightarrow \mathbb{R}$
# * $\sigma_\text{apply} : \mathbb{R}^{n \times p} \rightarrow \mathbb{R}^{n \times p}$
# * $S \in \mathbb{R}^{n \times p}$
```

``` {.python}
from sympy import Function

# Nvec = Symbol('N', commutative=False)

sigma = Function('sigma')
sigma(N[0,0])
```

```{=latex}
$\displaystyle \sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)}$
```
``` {.python}
# way 1 of declaring S
S = N.applyfunc(sigma); S
```

```{=latex}
$\displaystyle \left[\begin{matrix}\sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} & \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\\sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} & \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\\sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} & \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right]$
```
``` {.python}
# way 2 of declaring S (better way)
sigmaApply = lambda matrix:  matrix.applyfunc(sigma)

sigmaApply(N)
```

```{=latex}
$\displaystyle \left[\begin{matrix}\sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} & \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\\sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} & \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\\sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} & \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right]$
```
``` {.python}
sigmaApply(X**2) # can apply this function to any matrix argument.
```

```{=latex}
$\displaystyle \left[\begin{matrix}\sigma{\left(x_{11}^{2} + x_{12} x_{21} + x_{13} x_{31} \right)} & \sigma{\left(x_{11} x_{12} + x_{12} x_{22} + x_{13} x_{32} \right)} & \sigma{\left(x_{11} x_{13} + x_{12} x_{23} + x_{13} x_{33} \right)}\\\sigma{\left(x_{11} x_{21} + x_{21} x_{22} + x_{23} x_{31} \right)} & \sigma{\left(x_{12} x_{21} + x_{22}^{2} + x_{23} x_{32} \right)} & \sigma{\left(x_{13} x_{21} + x_{22} x_{23} + x_{23} x_{33} \right)}\\\sigma{\left(x_{11} x_{31} + x_{21} x_{32} + x_{31} x_{33} \right)} & \sigma{\left(x_{12} x_{31} + x_{22} x_{32} + x_{32} x_{33} \right)} & \sigma{\left(x_{13} x_{31} + x_{23} x_{32} + x_{33}^{2} \right)}\end{matrix}\right]$
```
``` {.python}
S = sigmaApply(v(X,W)) # composing
S
```

```{=latex}
$\displaystyle \left[\begin{matrix}\sigma{\left(w_{11} x_{11} + w_{21} x_{12} + w_{31} x_{13} \right)} & \sigma{\left(w_{12} x_{11} + w_{22} x_{12} + w_{32} x_{13} \right)}\\\sigma{\left(w_{11} x_{21} + w_{21} x_{22} + w_{31} x_{23} \right)} & \sigma{\left(w_{12} x_{21} + w_{22} x_{22} + w_{32} x_{23} \right)}\\\sigma{\left(w_{11} x_{31} + w_{21} x_{32} + w_{31} x_{33} \right)} & \sigma{\left(w_{12} x_{31} + w_{22} x_{32} + w_{32} x_{33} \right)}\end{matrix}\right]$
```
