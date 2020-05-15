import numpy as np
import matplotlib.pyplot as plt


class Iterator:
    def __init__(self, fct):
        self.function = fct
        self.x = None

    def iterate(self):
        for k in np.arange(1, 30):
            self.function.next()
        self.x = np.zeros(6)
        self.x[0] = self.function.next()
        this = self.function.next()
        k = 1
        while (self.x[0] - this) > np.exp(-3*np.log(10)):
            self.x[k] = this
            k += 1


class Function:
    def __init__(self, function):
        self.function = function
        self.first = 0.5
        self.x = None

    def next(self):
        if self.x is None:
            self.x = self.first
        else:
            self.x = self.function(self.x)
        return self.x

    def set_initial(self, x_initial):
        self.x = x_initial


class Visualize:
    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def plot(self, x, y):
        self.ax.plot(x, y)
        plt.show()

