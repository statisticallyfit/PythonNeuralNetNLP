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
# Markdown section title 1

Some **static** Markdown text.
"""

# %% markdown
# # This is a Big Title
# %% codecell
print("hi")
print(1987+1)
# %% codecell
print("Hello world!")
# %% markdown {input=True, eval=True, echo=True}
# Just testing that this markdown bit works:
# $$
# \int_a^b \text{sin}(\theta) \text{cos}(\theta) d\theta
# $$
# %% codecell {input=True, eval=True, echo=True}
# Render LaTex
import sympy as sp
x, y, z = sp.symbols('x y z')
x
# %% codecell {input=True, eval=True, echo=True}
y
# %% codecell
z
# %% codecell {input=True, eval=True, echo=True}
f = sp.sin(x * y) + sp.cos(y * z)
f
# %% codecell
sp.integrate(f, x)

# %% markdown
# That was an integral expression!

# %% codecell
import numpy as np
t = np.linspace(0, 20, 500)
t
