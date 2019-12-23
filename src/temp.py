# %% markdown
# # This is a Big Title
# %% codecell
print("hi")
print(1987+1)

# %% codecell
print("Hello world!")

# %% markdown
# Just testing that this markdown bit works:
# $$
# \int_a^b \text{sin}(\theta) \text{cos}(\theta) d\theta
# $$
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

# %% markdown
# That was an integral expression!

# %% codecell
import numpy as np
t = np.linspace(0, 20, 500)
t
