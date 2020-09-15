
# %% codecell
# TODO see here: https://stackoverflow.com/a/41581696

from sympy import *
A = Matrix(symarray('a', (4, 5))); A
# %% codecell
B = Matrix(symarray('b', (5, 3))); B
# %% codecell
C = A*B
C
# %% codecell
C.diff(A[1, 2])
# %% codecell
