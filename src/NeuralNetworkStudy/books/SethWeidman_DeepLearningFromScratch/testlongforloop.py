# %% codecell
#from IPython.display import display, Math, Latex
#from IPython.core.display import display_html
from sympy import *
#init_session(quiet=True)
#init_printing()
x = symbols('x')
arg = x
for i in range(120):
    arg += x**i
arg
