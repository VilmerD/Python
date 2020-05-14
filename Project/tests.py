from Matricies.nonlinear_matricies import *
from Newton.newton import newton
from Newton.preconditioners import *
import matplotlib.pyplot as plt
from time import time


n = 2 ** 14 - 1
t1 = time()
residuals, sols, etas, nits = newton(F, J, n, M=multigrid)
print(time() - t1)
fix, ax = plt.subplots()
plt.semilogy(abs(np.array(residuals)))
plt.show()


