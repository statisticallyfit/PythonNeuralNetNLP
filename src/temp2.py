"""
---
kernels-map:
  py: nn
jupyter:
  kernelspec:
    display_name: nn
    language: python
    name: nn
pandoctools:
  out: "*.pdf"
  # out: "*.ipynb"
...
"""

# %% codecell
# Render LaTex
import sympy as sp
x, y, z = sp.symbols('x y z')
x
# %% codecell
y
# %% codecell
z
# %% codecell
f = sp.sin(x * y) + sp.cos(y * z)
f
# %% codecell
sp.integrate(f, x)
