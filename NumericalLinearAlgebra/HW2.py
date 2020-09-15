import numpy.linalg as linalg
import scipy.linalg as slin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets as w
from NumericalLinearAlgebra.Orthogonalization import Orthogonalization


def problem2(A):
    OR = Orthogonalization(A)
    OR.grahm_Schmidt()
    Q = OR.Q
    R = OR.R
    print("Norm of Q: {}".format(linalg.norm(Q)))
    print("How unorthogonal they are: {}".format(linalg.norm(np.identity(n) - Q.T.dot(Q))))
    print("det: 1 + {}".format(np.linalg.det(Q.T.dot(Q)) - 1))


def problem3(A):
    Q, R = linalg.qr(A)
    print("Norm of Q: {}".format(linalg.norm(Q)))
    print("How unorthogonal they are: {}".format(linalg.norm(np.identity(n) - Q.T.dot(Q))))
    print("det: 1 + {}".format(np.linalg.det(Q.T.dot(Q))- 1))


def problem4(A):
    OR = Orthogonalization(A)
    R, Q = OR.householder()
    print("A:\n {}".format(A))
    print("QR:\n {}".format(Q * R.dot(np.identity(2))))
    Q1 = Q * np.identity(3)
    print("Norm of I - Q^TQ:\n {}".format(linalg.norm((np.identity(3) - Q1.dot(Q1.T)))))

    sQ, sR = slin.qr(A)

    print("Our QR:\nQ:\n {}\nR:\n {}".format(Q1, R))
    print("Scipy QR:\nQ:\n {}\nR:\n {}".format(sQ, sR))


def problem6():
    A = plt.imread('kvinna.jpg')
    U, S, VT = slin.svd(A)
    sigmamax = max(S)
    fig, ax = plt.subplots()

    slider_ax = plt.axes([0.3, 0.15, 0.4, 0.01])
    s = w.Slider(slider_ax, "Maximum sigma", 0, sigmamax, valinit=100)


def compremize(A, maximum_sigma):
    U, S, VT = slin.svd(A)
    S_nu = np.diag(S)
    S_nu[S < maximum_sigma] = 0
    A_nu = U.dot(S_nu.dot(VT))
    return A_nu


problem6()