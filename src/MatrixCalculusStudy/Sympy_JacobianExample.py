# %% codecell
from sympy import Matrix, MatrixSymbol, Symbol, derive_by_array



X = Matrix(MatrixSymbol('x', 3,3)); X
W = Matrix(MatrixSymbol('w', 3,2)); W
# %% codecell
X*W

# %% codecell
derive_by_array(X*W, X)
# %% codecell
(X*W).diff(X)




# %% codecell
from sympy import diff, sin, exp, symbols, Function
#from sympy.core.multidimensional import vectorize #

#@vectorize(0,1)
#def vdiff(func, arg):
#     return diff(func, arg)



x, y, z = symbols('x y z')
f, g, h = list(map(Function, 'fgh'))

xv = x,y,z
#f(xv).subs({x:1, y:2,z:3})
yv = [f(*xv), g(*xv), h(*xv)]; yv
# %% codecell
Matrix(yv)
# %% codecell
Matrix(xv)
# %% codecell
Matrix(yv).jacobian(xv)



# %% codecell
derive_by_array(yv, xv)

# %% codecell
assert Matrix(derive_by_array(yv, xv)).transpose() == Matrix(yv).jacobian(xv)

# %% codecell

### TEST 2: substituting values
m = Matrix(yv).jacobian(xv)
m.subs({x:1, y:2, z:3})

# %% codecell
m.subs({f(*xv):x**2 * y*z, g(*xv):sin(x*y*z*3), h(*xv):y + z*exp(x)})

# %% codecell
m_subs = m.subs({f(*xv):x**2 * y*z, g(*xv):sin(x*y*z*3), h(*xv):y + z*exp(x)})
m_subs.doit()

# %% codecell
m_subs.doit().subs({x:1, y:2, z:3})
# %% codecell
