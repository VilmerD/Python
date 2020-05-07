from Bifurcasion import iterator as s
import numpy as np
import matplotlib.pyplot as plt

fct = s.Function(lambda x: 3.8*x*(1 - x))
fct.set_initial(0.4)

It = s.Iterator(fct)
It.iterate()

plotter = s.Visualize()
plotter.plot(np.arange(0, len(It.x)), It.x)