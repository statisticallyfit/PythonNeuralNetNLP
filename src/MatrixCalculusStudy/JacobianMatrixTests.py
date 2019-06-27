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




#
# `Jacobian matrix` of a `scalar function` is the `function gradient`
#
def testJacobianMatrixOfScalarFunctionIsFunctionGradient():

    x1s = np.linspace(start=10.0, stop=20.0, num=333, dtype=X1T)  #array
    x2s = np.linspace(start=20.0, stop=45.0, num=333, dtype=X2T) # array
    x3s = np.linspace(start=30.0, stop=98.0, num=333, dtype=X2T) # array
    xs = np.array(list(zip(x1s, x2s, x3s))) # matrix
        #np.array([x1s, x2s, x3s]) #

    jb = nd.Jacobian(multivarScalarFunction)

    # Apply the Jacobian function to all 3 dimensional points in the linear space
    # There are 2 variables in the function and so the jacobian has the form:
    # ( df1 / dx1   df1 / dx2 ) ????????????????????????????????????????????????????????
    # ------ ( df2 / dx1   df2 / dx2 )
    # ------ ( df2 / dx1   df2 / dx2 )
    for x in xs:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        jac = jb([x1, x2, x3])
        grad = nd.Gradient(multivarScalarFunction)([x1, x2, x3])
        assert np.allclose(jac, grad) == True


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


def exp1():
    x = np.linspace(-2, 2, 100)

    for i in range(10):
        df = nd.Derivative(np.tanh, n=i)
        y = df(x)
        h = plt.plot(x, y / np.abs(y).max())

    plt.show()


if __name__ == "__main__":
    main()
