import numpy as np
import scipy.integrate as int
import matplotlib.pyplot as plt
from matplotlib import widgets as w


def system(t, y):
    t0 = 5
    (i, s, r) = y

    di = transmission_rate(t) * i * s - k0 * i
    dr = k0 * i
    ds = - dr - di
    return [di, ds, dr]


def solve_it():
    t_eval = np.linspace(0, t_max, 200)
    solution = int.solve_ivp(system, [0, t_max], y0, dense_output=True, t_eval=t_eval)
    y = solution.y
    t = solution.t
    return t, y


def transmission_rate(t):
    if t < transmission_rate.t0:
        return transmission_rate.tr0
    else:
        return transmission_rate.tr0 * transmission_rate.q


def make_toggler(line):
    def toggle(val):
        line.toggled = not line.toggled
        line.set_visible(line.toggled)
    return toggle


N = 1
k0 = 1
transmission_rate.tr0 = 2
transmission_rate.q = 1/4
transmission_rate.t0 = 5

t_max = 20

y0 = (0.01, N, 0)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.4, bottom=0.25)
t, y = solve_it()
(i, s, r) = y
lines = plt.plot(t, i, 'r', t, s, 'b', t, r, 'g')

q_ax = plt.axes([0.3, 0.15, 0.4, 0.01])
tr_ax = plt.axes([0.3, 0.1, 0.4, 0.01])
bp_ax = plt.axes([0.3, 0.05, 0.4, 0.01])
bp_slider = w.Slider(bp_ax, "Breakpoint", 0, t_max, valinit=5)
tr_slider = w.Slider(tr_ax, "Transmission", 0, 10, valinit=2)
q_slider = w.Slider(q_ax, "Quarantine quotient", 0, 1, valinit=1/4)

red_ax = plt.axes([0.02, 0.4, 0.3, 0.04])
blue_ax = plt.axes([0.02, 0.45, 0.3, 0.04])
green_ax = plt.axes([0.02, 0.5, 0.3, 0.04])
red_button = w.Button(red_ax, "Infected", color='tab:red')
blue_button = w.Button(blue_ax, "Suseptible", color='tab:blue')
green_button = w.Button(green_ax, 'Recovered', color='tab:green')
Buttons = (red_button, blue_button, green_button)

for k in range(0, 3):
    lines[k].toggled = True
    Buttons[k].on_clicked(make_toggler(lines[k]))


def bp(val):
    t0 = bp_slider.val
    tr0 = tr_slider.val
    q = q_slider.val
    transmission_rate.t0 = t0
    transmission_rate.q = q
    transmission_rate.tr0 = tr0
    t, y = solve_it()
    for k in range(0, 3):
        lines[k].set_ydata(y[k])


tr_slider.on_changed(bp)
q_slider.on_changed(bp)
bp_slider.on_changed(bp)
plt.show()


