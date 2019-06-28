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


# The rosenbrock function
def multivarScalarFunction(x: np.array) -> np.double :
    return (1 - x[0]) ** 2 + 105. * (x[1] - x[0] ** 2) ** 2

# the partial derivative with respect to x of the multivariate scalar Rosenbrock function
def dfdx(x: np.double, y: np.double) -> np.double :
    return


#
# `Jacobian matrix` of a `scalar function` is the `function gradient`
#
def testJacobianMatrixOfScalarFunctionIsFunctionGradient():

    x1s = np.linspace(start=10.0, stop=20.0, num=333, dtype=np.float64)  #array
    x2s = np.linspace(start=20.0, stop=45.0, num=333, dtype=np.float64) # array
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


# #
# # #todo vector-valued functions
# https://numdifftools.readthedocs.io/en/latest/tutorials/getting_started.html
# #
# def testJacobianMatrixOfVectorValuedFunction():
#     x1s = [2, 3, 4]
#     x2s = [1, 4, 9]
#     jac = nd.Jacobian(rosen)((x1s, x2s))
#     grad1 = nd.Gradient(rosen)(x1s)
#     grad2 = nd.Gradient(rosen)(x2s)
#
#     assert np.allclose(jac, (grad1, grad2)) == True


# Case 3: jacobian matrix of a nonlinear transformation of variables (for integration, thomas book)

def exp1():
    x = np.linspace(-2, 2, 100)

    for i in range(10):
        df = nd.Derivative(np.tanh, n=i)
        y = df(x)
        h = plt.plot(x, y / np.abs(y).max())

    plt.show()


if __name__ == "__main__":
    main()
