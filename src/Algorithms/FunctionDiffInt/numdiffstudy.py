import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt


def main():
    exp1()
    exp_jacobian_01()
    exp_jacobian_02()


def rosen(x):
    return (1 - x[0]) ** 2 + 105. * (x[1] - x[0] ** 2) ** 2


def exp_jacobian_01():
    grad = nd.Gradient(rosen)([1, 1])
    assert np.allclose(grad, 0) == True


#
# `Jacobian matrix` of a `scalar function` is the `function gradient`
#
def exp_jacobian_02():
    jac = nd.Jacobian(rosen)([2, 3])
    grad = nd.Gradient(rosen)([2, 3])
    assert np.allclose(jac, grad) == True


def exp1():
    x = np.linspace(-2, 2, 100)

    for i in range(10):
        df = nd.Derivative(np.tanh, n=i)
        y = df(x)
        h = plt.plot(x, y / np.abs(y).max())

    plt.show()


if __name__ == "__main__":
    main()
