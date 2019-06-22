import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
#from typing import  *


def main():
    #exp1()
    #exp_jacobian_01()
    exp_jacobian_02()
    # exp_jacobian_03()


def rosen(x : np.ndarray) -> np.double:
    return (1 - x[0]) ** 2 + 105. * (x[1] - x[0] ** 2) ** 2


def exp_jacobian_01():
    grad = nd.Gradient(rosen)([1, 1])
    assert np.allclose(grad, 0) == True


X1T = np.float64
X2T = np.float64
X3T = np.float64


#
# `Jacobian matrix` of a `scalar function` is the `function gradient`
#
def exp_jacobian_02():

    x1s = np.linspace(start=10.0, stop=20.0, num=333, dtype=X1T)
    x2s = np.linspace(start=20.0, stop=45.0, num=333, dtype=X2T)
    x3s = np.linspace(start=30.0, stop=98.0, num=333, dtype=X2T)
    xs = np.array(list(zip(x1s, x2s, x3s)))
        # np.array(
        # np.array(list(zip(x1s, x2s, x3s)))
        # dtype=np.dtype([('x1', X1T), ('x2', X2T), ('x3', X3T)]))

    jb = nd.Jacobian(rosen)

    # Apply the Jacobian function to all 3 dimensional points in the linear space
    for x in xs:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        jac = jb([x1, x2, x3])
        grad = nd.Gradient(rosen)([x1, x2, x3])
        assert np.allclose(jac, grad) == True


# #
# #
# #
# def exp_jacobian_03():
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
