import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt


def main():
    x = np.linspace(-2, 2, 100)

    for i in range(10):
        df = nd.Derivative(np.tanh, n=i)
        y = df(x)
        h = plt.plot(x, y/np.abs(y).max())

    plt.show()

if __name__ == "__main__":
    main()


