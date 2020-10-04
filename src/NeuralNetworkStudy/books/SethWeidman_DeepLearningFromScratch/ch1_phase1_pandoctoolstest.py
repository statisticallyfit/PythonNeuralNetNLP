"""
---
pandoctools:
  profile: Kiwi
  out: "*.md"
  # out: "*.pdf"
input: False
eval: True
echo: True
error: raise
...
"""


# %%

from IPython.display import Markdown

from sympy import Matrix, Symbol, derive_by_array, Lambda, symbols, Derivative, diff
from sympy.abc import x, y, i, j, a, b



# %% markdown
# Defining variable-element matrices $X \in \mathbb{R}^{n \times m}$ and $W \in \mathbb{R}^{m \times p}$:

# %%
def var(letter: str, i: int, j: int) -> Symbol:
    letter_ij = Symbol('{}_{}{}'.format(letter, i+1, j+1), is_commutative=True)
    return letter_ij


n,m,p = 3,3,2

X = Matrix(n, m, lambda i,j : var('x', i, j)); X
# %%
W = Matrix(m, p, lambda i,j : var('w', i, j)); W

# %% markdown
# Defining $N = \nu(X, W) = X \times W$
#
# * $\nu : \mathbb{R}^{(n \times m) \times (m \times p)} \rightarrow \mathbb{R}^{n \times p}$
# * $N \in \mathbb{R}^{n \times p}$

# %%
v = Lambda((a,b), a*b); v

# %%
N = v(X, W); N

# %% markdown
# Defining $S = \sigma_{\text{apply}}(N) = \sigma_{\text{apply}}(\nu(X,W)) = \sigma_\text{apply}(X \times W) = \Big \{ \sigma(XW_{ij}) \Big\}$.
#
#
# Assume that $\sigma_{\text{apply}} : \mathbb{R}^{n \times p} \rightarrow \mathbb{R}^{n \times p}$ while $\sigma : \mathbb{R} \rightarrow \mathbb{R}$, so the function $\sigma_{\text{apply}}$ takes in a matrix and returns a matrix while the simple $\sigma$ acts on the individual elements $N_{ij} = XW_{ij}$ in the matrix argument $N$ of $\sigma_{\text{apply}}$.
#
# * $\sigma : \mathbb{R} \rightarrow \mathbb{R}$
# * $\sigma_\text{apply} : \mathbb{R}^{n \times p} \rightarrow \mathbb{R}^{n \times p}$
# * $S \in \mathbb{R}^{n \times p}$

# %%
from sympy import Function

# Nvec = Symbol('N', commutative=False)

sigma = Function('sigma')
sigma(N[0,0])

# %%
# way 1 of declaring S
S = N.applyfunc(sigma); S

# %%
# way 2 of declaring S (better way)
sigmaApply = lambda matrix:  matrix.applyfunc(sigma)

sigmaApply(N)

# %%
sigmaApply(X**2) # can apply this function to any matrix argument.
# %%
S = sigmaApply(v(X,W)) # composing
S
