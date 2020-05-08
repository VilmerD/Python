import numpy as np
import scipy.sparse.linalg as splin
import scipy.sparse as sp
import scipy.linalg as slin
import matplotlib.pyplot as plt


def interval(n, L=1):
    dx = L / (n + 1)
    return (np.arange(1, n + 1) * dx).reshape((n, ))


def jacobi(A, D, x, b, w):
    return x - w * splin.spsolve(D, A.dot(x) - b)


def njacobi(A, D, x, b, n, w=1):
    for k in range(0, n):
        x = jacobi(A, D, x, b, w)
    return x


def interpolator(v):
    n = len(v)
    u = np.zeros(n + 2)
    u[1: -1] = v.reshape(n)
    v_new = np.zeros(2 * n + 1)

    for k in np.arange(0, n):
        v_new[2 * k] = (u[k + 1] + u[k]) / 2
        v_new[2 * k + 1] = u[k + 1]
    v_new[-1] = v[-1] / 2
    return v_new


def restrictor(v):
    n = len(v)
    if n > 1:
        vnew = np.zeros(int((n - 1) / 2))
        for k in np.arange(0, int((n - 1) / 2)):
            vnew[k] = (v[2 * k] + 2 * v[2 * k + 1] + v[2 * k + 2]) / 4
        return vnew


def twogrid(Afun, b, x0=None, pre=1, post=1):
    n = len(b)
    n_l = int((n - 1) / 2)
    A = Afun(n)
    D = sp.csr_matrix(sp.diags(A.diagonal()))
    if x0 is None:
        x0 = np.zeros((n, ))

    xtilde = njacobi(A, D, x0, b, 1, pre)
    x_biss = A.dot(xtilde)
    r = x_biss - b
    rl_1 = restrictor(r)
    el_1 = splin.spsolve(Afun(n_l), rl_1)
    xtilde = xtilde - interpolator(el_1)
    xtilde = njacobi(A, D, xtilde, b, 1, post)

    return xtilde


def discrete_second(n, L=1):
    dx = L / (n + 1)
    return sp.csr_matrix(dx ** -2 * sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n)))


def source(n):
    x = interval(n)
    return 4 * np.pi ** 2 * np.sin(np.pi * x ** 2).reshape((n,))


def figs():
    ns = [63, 127]
    pres = [1, 2]
    posts = [1, 2]
    fig, ax = plt.subplots()
    axes = []
    for k in [0, 1]:
        n = ns[k]
        x = twogrid(discrete_second, source(n), pre=pres[0], post=posts[0])
        axes.append(plt.subplot(2, 2, 2 * k + 1))
        uxx = discrete_second(n).dot(x)
        r = uxx + source(n)
        plt.plot(interval(n), r)

        x = twogrid(discrete_second, source(n), pre=pres[1], post=posts[1])
        axes.append(plt.subplot(2, 2, 2 * k + 2))
        uxx = discrete_second(n).dot(x)
        r = uxx + source(n)
        plt.plot(interval(n), r)

        axes[2*k].set_ylabel("n = {}".format(n))

    axes[0].set_title("1 pre- and postsmoother")
    axes[1].set_title("2 pre- and postsmoothers")
    plt.suptitle("Residuals")
    plt.show()


def control():
    n = 127
    x = interval(n).reshape((n, ))
    y = twogrid(lambda u: - discrete_second(u), np.pi ** 2 * np.sin(np.pi*x), pre=1, post=1)

    uxx = discrete_second(n).toarray().dot(y)
    res = uxx - np.pi ** 2 * np.sin(np.pi * x)
    sol = np.sin(np.pi * x)
    e = sol - y

    fig, ax = plt.subplots()
    ax1 = plt.subplot(121)
    plt.plot(x, res, 'y')
    ax2 = plt.subplot(122)
    plt.plot(x, e, 'r')
    plt.suptitle("Residual and error for the 2-grid method for control problem")
    ax1.set_title("Residual")
    ax2.set_title("True error")
    plt.show()
