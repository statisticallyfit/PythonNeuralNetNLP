# %% codecell
from sympy import diff, sin, exp, symbols, Function, Matrix, FunctionMatrix, derive_by_array
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
# ### for deriv of scalar-valued multivariate function with respect to the vector
# Definition from Helmut:
arg = Matrix([*xv]); arg
# %% codecell
f(*xv).diff(arg)

# %% codecell
derive_by_array(f(*xv), arg)


# %% codecell
# ### for deriv of a vector-valued function by its scalar argument
yv = [f(x), g(x), h(x)]; yv
Matrix(yv)

# %% codecell
Matrix(yv).diff(x) # incorrect shape (is column-wise, must be row-wise like below)
# %% codecell
derive_by_array(yv, x) # Correct shape (row-wise)
# %% codecell
assert Matrix(derive_by_array(yv, x)).transpose() == Matrix(yv).transpose().diff(x)



# %% codecell
# ### for vector chain rule
