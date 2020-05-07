from Multigrid.multigrid import v_cycle, v_cycle_explicit, full_multigrid
from Multigrid.matricies import poisson_matrix_1d, interpolator2d, restrictor2d
from Multigrid.gauss_seidel import gauss_seidel, n_gauss_seidel
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from numpy import pi, sin, e, cos
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.sparse import linalg

import time


def test_recursive():
    N = 1023

    x = np.arange(1, N + 1) / (N + 1)
    v = sin(13 * pi * x) / 2
    f = - 1000 * e ** (- 10000 * (x - 0.5)**2) + 1000 * e ** (- 10000 * (x - 0.25) ** 2)

    v = v_cycle(v, f)
    fig, ax = plt.subplots()
    plt.plot(x, v)
    plt.show()


def test_all():
    N = 63

    x = np.arange(1, N + 1) / (N + 1)
    v0 = sin(3 * pi * x) / 2
    f = - pi ** 2 * sin(pi * x)
    A = poisson_matrix_1d(N)
    v0 = v_cycle(v0, f)

    fig, ax = plt.subplots()
    plt.plot(x, v0)
    plt.show()


def atest():
    n = 63

    x = np.arange(1, n + 1) / (n + 1)
    v = np.sin(12*x*np.pi).reshape(n, 1)/3 + np.sin(3*x*np.pi).reshape(n, 1)
    f = np.zeros(n).reshape(n, 1)
    f = -(3*np.pi)**2*np.sin(3*x*np.pi).reshape(n, 1)

    c = np.concatenate(([-2, 1], np.zeros(n - 2)))
    a = poisson_matrix_1d(n)

    v1 = v.copy()
    for k in np.arange(1, 3):
        gauss_seidel(a, v, f)

    v2 = v.copy()
    for k in np.arange(1, 4):
        gauss_seidel(a, v, f)

    v3 = v.copy()
    for k in np.arange(1, 20):
        gauss_seidel(a, v, f)

    fig, ax = plt.subplots()
    plt.subplot(141)
    plt.axis([0, 1, -1.4, 1.4])
    plt.plot(x, v1)
    plt.subplot(142)
    plt.axis([0, 1, -1.4, 1.4])
    plt.plot(x, v2)
    plt.subplot(143)
    plt.axis([0, 1, -1.4, 1.4])
    plt.plot(x, v3)
    plt.subplot(144)
    plt.axis([0, 1, -1.4, 1.4])
    plt.plot(x, v)
    plt.show()


def twotest():
    n = 255

    x = np.arange(1, n + 1) / (n + 1)
    v = np.sin(5*np.pi * x)
    v0 = np.copy(v)
    f = -(3*np.pi) ** 2 * np.sin(3*np.pi*x)
    sol = np.sin(3*np.pi*x)

    # VCycle
    t0 = time.time()
    v1 = v_cycle_explicit(f, v, 2)
    t1 = time.time()
    t_cycle = t1 - t0
    t_cycle = round(t_cycle*1000)

    # True solution
    t0 = time.time()
    v3 = linalg.spsolve(poisson_matrix_1d(n), f)
    t1 = time.time()
    t_true = t1 - t0
    t_true = round(t_true*1000)


    # Initializing plot
    fig, ax = plt.subplots()

    plt.subplot(131)
    plt.title("Initial error")
    plt.axis([0, 1, -2, 2])
    plt.plot(x, v0 - sol)

    plt.subplot(132)
    title1 = "({}ms) V-cycle".format(t_cycle)
    plt.title(title1)
    plt.axis([0, 1, -2, 2])
    plt.plot(x, v1 - sol)

    plt.subplot(133)
    title3 = "({}ms) true solution".format(t_true)
    plt.title(title3)
    plt.axis([0, 1, -2, 2])
    plt.plot(x, v3 - sol)

    plt.show()


def timing():
    ntimes = 100
    times = np.zeros(ntimes)
    N = 255
    x = np.arange(1, N + 1) / (N + 1)
    v = np.sin(12 * x * np.pi).reshape(N, ) / 3 + np.sin(np.pi * x).reshape(N, )
    f = -(3 * np.pi) ** 2 * np.sin(3 * x * np.pi).reshape(N, )
    for k in np.arange(0, ntimes):
        t0 = time.time()
        v_cycle_explicit(f, v, 2)
        times[k] = t0 - time.time()

    fig, ax = plt.subplots()
    plt.hist(times)
    plt.show()


def threetest():
    n = 127
    x = np.arange(1, n + 1) / (n + 1)
    f0 = 5
    delta_f = 1
    a0 = 10000
    delta_a = 1000
    shift = 9
    delta_shift = 0.1

    func = "Pr"

    def fun(f, a, s):
        return - e ** (s - a * sin(f * pi * (x - 0.5)) ** 4) +\
               e ** (s - a * 10 * sin(f * pi * (x - 0.5)) ** 90) - 4100

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    f = fun(f0, a0, shift)
    v1 = full_multigrid(f, 1, fun)

    plt.subplot(121)
    l, = plt.plot(x, v1)
    plt.subplot(122)
    r, = plt.plot(x, f)

    amp_ax = plt.axes([0.2, 0.15, 0.65, 0.03])
    freq_ax = plt.axes([0.2, 0.1, 0.65, 0.03])
    shit_ax = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider_f = Slider(freq_ax, "Frekvens", 1, 15, valinit=f0, valstep=delta_f)
    slider_a = Slider(amp_ax, "Amplitud", 0, 30000, valinit=a0, valstep=delta_a)
    slider_shift = Slider(shit_ax, "Shift", 0, 15, valinit=shift, valstep=delta_shift)

    def recalculate(val):
        frequency = slider_f.val
        a = slider_a.val
        s = slider_shift.val

        f_new = fun(frequency, a, s)
        v = full_multigrid(f_new, 1, "1")

        l.set_ydata(v)
        r.set_ydata(f_new)
        fig.canvas.draw_idle()
    slider_f.on_changed(recalculate)
    slider_a.on_changed(recalculate)
    slider_shift.on_changed(recalculate)
    plt.show()


def twodtest():
    dim = "2"
    n = 63

    x = np.arange(1, n + 1) / (n + 1)
    xx, yy = np.meshgrid(x, x)
    f = ((3 * pi) ** 2 + (5 * pi) ** 2) * sin(3 * pi * xx) * sin(5 * pi * yy)
    sol = - sin(3 * pi * xx) * sin(5 * pi * yy)

    v = full_multigrid(f.reshape(n**2), 1, dim)

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    _ = ax1.plot_surface(xx, yy, v.reshape((n, n)))
    ax2 = fig.add_subplot(122, projection='3d')
    _ = ax2.plot_surface(xx, yy, v.reshape((n, n)) - sol)
    plt.show()


def twodtest2():
    dim = "2"
    n = 63
    m = int((n + 1) / 2)

    x = np.arange(1, n + 1) / (n + 1)
    xx, yy = np.meshgrid(x, x)
    f = - 23000 * e ** (-10000 * (xx - 0.5) ** 2 - 10000 * (yy - 0.5) ** 2)
    sol = - np.log(((xx - 0.5) ** 2 + (yy - 0.5) ** 2)) / 2

    v = full_multigrid(f.reshape(n ** 2), 1, dim)

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    _ = ax1.plot_surface(xx, yy, v.reshape((n, n)))
    ax2 = fig.add_subplot(122, projection='3d')
    _ = ax2.plot_surface(xx, yy, sol)
    plt.show()


def restrictortest():
    n = 7
    x = np.arange(1, n + 1) / (n + 1)
    xx, yy = np.meshgrid(x, x)

    v = sin(pi * xx).reshape(n**2)

    fig = plt.figure()
    fig.n = 7
    fig.v = v
    ax = fig.add_subplot(111, projection='3d')
    s = ax.plot_surface(xx, yy, v.reshape((n, n)))

    iter_ax = plt.axes([0.1, 0.05, 0.2, 0.05])
    iter_button = Button(iter_ax, "Iterate")
    res_ax = plt.axes([0.3, 0.05, 0.2, 0.05])
    res_button = Button(res_ax, "Restrict")
    cycle_ax = plt.axes([0.5, 0.05, 0.2, 0.05])
    cycle_button = Button(cycle_ax, "Cycle")

    def restrict(val):
        fig.n = int((fig.n - 1) / 2)
        fig.v = restrictor2d(fig.v)
        xs = np.arange(1, fig.n + 1) / (fig.n + 1)
        xx, yy = np.meshgrid(xs, xs)
        ax.cla()
        ax.plot_surface(xx, yy, fig.v.reshape((fig.n, fig.n)))

    def interpolate(val):
        fig.n = int((2 * fig.n + 1))
        fig.v = interpolator2d(fig.v)
        xl = np.arange(1, fig.n + 1) / (fig.n + 1)
        xx, yy = np.meshgrid(xl, xl)
        ax.cla()
        ax.plot_surface(xx, yy, fig.v.reshape((fig.n, fig.n)))

    def cycle(val):
        xl = np.arange(1, fig.n + 1) / (fig.n + 1)
        xx, yy = np.meshgrid(x, x)
        f = sin(3 * pi * xx) * e ** (-100 * (yy - 0.5) ** 2)
        fig.v = v_cycle(fig.v, f.reshape(fig.n ** 2), "2")
        ax.cla()
        ax.plot_surface(xx, yy, fig.v.reshape((fig.n, fig.n)))
    iter_button.on_clicked(interpolate)
    res_button.on_clicked(restrict)
    cycle_button.on_clicked(cycle)
    plt.show()


threetest()
