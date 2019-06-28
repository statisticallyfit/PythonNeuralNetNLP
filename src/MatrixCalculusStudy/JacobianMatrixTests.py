import numpy as np
import xarray as xr
import numdifftools as nd
import matplotlib.pyplot as plt


# example data array
#theArray = xr.DataArray([11,22,33], dims=['x'], coords={'x': [0,1,2]})
#    print(theArray)
#    print("\n")
#    print("first: " + theArray.x0)
#    print("second: " + theArray.x1)
#    print("third: " + theArray.x2)

X1T = np.float64
X2T = np.float64
X3T = np.float64




def main():
    testJacobianMatrixOfScalarFunctionIsFunctionGradient()
    # exp_jacobian_03()


# Testing ---------------------------------------------------------------------------------------------------


# The rosenbrock function
def multivarScalarFunction(x: np.array) -> np.double :
    return (1 - x[0]) ** 2 + 105. * (x[1] - x[0] ** 2) ** 2

# Returns: the partial derivative with respect to x, of the rosenbrock function evaluated at numbers xn, yn
def dfdx(xn: np.double, yn: np.double) -> np.double :
    return (420*(xn**3 - xn*yn) + 2*xn - 2)

# Returns: the partial derivative with respect to y, of the rosenbrock function evaluated at numbers xn, yn
def dfdy(xn: np.double, yn: np.double) -> np.double :
    return (210*(yn - xn**2))




# `Jacobian matrix` of a `scalar function` is the `function gradient`
#
def testJacobianMatrixOfScalarFunctionIsFunctionGradient():

    x1s = np.linspace(start=10.0, stop=20.0, num=10, dtype=np.float64)  #array
    x2s = np.linspace(start=20.0, stop=45.0, num=10, dtype=np.float64) # array
    xs = np.array(list(zip(x1s, x2s))) # matrix , where x1s and x2s are columnwise

    jb = nd.Jacobian(multivarScalarFunction)

    # Apply the Jacobian function to all 2 dimensional points in the linear space
    # There are 2 variables in the function and so the jacobian has the form:
    # ( df / dx1   df / dx2 )
    # which is the same as the gradient vector!
    for row in xs:
        xn = row[0]
        yn = row[1]

        # Calculating the jacobian matrix, where xn and yn are inputs to all the component functions
        jacobianMatrix = jb([xn, yn])

        # Calculating the gradient vector, where xn and yn are inputs to all the component functions
        gradientVector = nd.Gradient(multivarScalarFunction)([xn, yn])

        # Testing that jacobian matrix (1 x 2) equals gradient vector
        assert np.allclose(jacobianMatrix, gradientVector) == True

        # Testing that Gradient components are just dfdx and dfdy evaluated at particular
        # values of x and y
        assert np.allclose(gradientVector[0], dfdx(xn, yn)) & np.allclose(gradientVector[1], dfdy(xn, yn))


def multivarVectorFunction(x: np.double, y: np.double, z: np.double) -> np.array :
    return [3*x**2 * y,         # f1(x,y,z)
            5*x + y**3 + z,     # f2(x,y,z)
            2*x*y*z + 6*x*y,    # f3(x,y,z)
            3*y*z**3,           # f4(x,y,z)
            4*x**2 *z**2]       # f5(x,y,z)



# Testing Case 2: Jacobian matrix of vector-valued function

# There are 3 variables and 5 component functions, so the Jacobian matrix has the form:
# (df1_dx   df1_dy   df1_dz)
# (df2_dx   df2_dy   df2_dz)
# (df3_dx   df3_dy   df3_dz)
# (df4_dx   df4_dy   df4_dz)
# (df5_dx   df5_dy   df5_dz)

# which is just the column-wise matrix of gradient vectors of the component functions

#  / GRAD f1(x,y,z) \
# | GRAD f2(x,y,z)  |
# | GRAD f3(x,y,z)  |
# | GRAD f4(x,y,z)  |
# \ GRAD f5(x,y,z) /

def testJacobianMatrixOfVectorValuedFunction():
    return 1 # todo start here tomorrow
# USE THESE:
# http://faculty.bard.edu/bloch/multivariablevectorvalued_notes_gray.pdf
# (matrix calc) https://explained.ai/matrix-calculus/index.html
# jacobian tests: https://github.com/pbrod/numdifftools/blob/5319098c39d9d41f7b945bfb6605ddbec7f6548f/src/numdifftools/tests/test_numdifftools.py
# examples: https://buildmedia.readthedocs.org/media/pdf/numdifftools/latest/numdifftools.pdf
# package details: https://numdifftools.readthedocs.io/en/latest/reference/numdifftools.html





# Testing Case 3: jacobian matrix of a nonlinear transformation of variables (for integration, thomas book)




def exp1():
    x = np.linspace(-2, 2, 100)

    for i in range(10):
        df = nd.Derivative(np.tanh, n=i)
        y = df(x)
        h = plt.plot(x, y / np.abs(y).max())

    plt.show()


if __name__ == "__main__":
    main()
