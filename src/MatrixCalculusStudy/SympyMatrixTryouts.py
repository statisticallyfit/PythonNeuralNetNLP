# %% codecell
from sympy import sin, cos, Matrix
from sympy.abc import rho, phi
X = Matrix([rho*cos(phi), rho*sin(phi), rho**2])
Y = Matrix([rho, phi])

X
# %% codecell
Y

# %% codecell
X.jacobian(Y)

# %% codecell
from sympy import MatrixSymbol, Matrix
from sympy.core.function import Function
from sympy import FunctionMatrix, Lambda

n, m, p = 3, 3, 2

X = MatrixSymbol('X', n, m)
W = MatrixSymbol('W', m, p)

(X.T*X).I*W


# %% codecell
Matrix(X)
# %% codecell
Matrix(X * W)




# %% codecell
from sympy import I, Matrix, symbols
from sympy.physics.quantum import TensorProduct

m1 = Matrix([[1,2],[3,4]])
m2 = Matrix([[1,0],[0,1]])

m1
# %% codecell
TensorProduct(m1, m2)

# %% codecell

g = FunctionMatrix(3,3, Lambda((i, j), (i,j)))

Matrix(g)
# %% codecell
g.as_explicit()

# %% codecell
f = Function('f')
F = FunctionMatrix(3,4, f)
Matrix(F)


# %% codecell
y = Function('y')(X)
y
# %% codecell
# y.diff(X)
from sympy.abc import x,y,z,t
from sympy import derive_by_array
from sympy import sin, exp, cos

#basis = Matrix([x, y, z])
Matrix(X)

m = Matrix([[exp(x), sin(y*z), t*cos(x*y)], [x, x*y, t], [x,x,z] ])
basis = [[x,y,z], [x,y,z], [x,y,z]]
ax = derive_by_array(m, basis)
ax

# %% codecell
f = Matrix([exp(x), sin(y*z), t*cos(x*y)])
v = Matrix([x,y,z])
f.jacobian(v)
# %% codecell
assert Matrix(ax).transpose() == f.jacobian(v)


# %% codecell

from sympy.abc import x
a = Function('a')
a.diff(x)
# %% codecell
a(x).diff(x)
# %% codecell
b = Function('b')
b.diff(x)
# %% codecell
b(x).diff(x)
# %% codecell
# h = a(b(x)) # Doesn't work when declaring the variable for a o rb functions 
h = a(b(x))
h
# %% codecell
h.diff(x)
