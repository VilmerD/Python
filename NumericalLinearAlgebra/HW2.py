import numpy.linalg as linalg
import scipy.linalg as slin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets as w
from NumericalLinearAlgebra.Orthogonalization import Orthogonalization


def problem2(A):
    m, n = A.shape
    OR = Orthogonalization(A)
    OR.grahm_Schmidt()
    Q = OR.Q
    print(Q.dot(Q.T))
    print("Norm of Q: {}".format(linalg.norm(Q)))
    print("How unorthogonal they are: {}".format(linalg.norm(np.identity(m) - Q.T.dot(Q))))
    print("det: 1 + {}".format(np.linalg.det(Q.T.dot(Q)) - 1))


def problem3(A):
    Q, R = linalg.qr(A)
    m, n = A.shape
    print("Norm of Q: {}".format(linalg.norm(Q)))
    print("How unorthogonal they are: {}".format(linalg.norm(np.identity(n) - Q.T.dot(Q))))
    print("det: 1 + {}".format(np.linalg.det(Q.T.dot(Q))- 1))


def problem4(A):
    m, n = A.shape
    OR = Orthogonalization(A)
    R, Q = OR.householder()

    Q1 = Q * np.identity(m)
    print("Norm of A - QR:\n {}".format(linalg.norm(A - Q.dot(R))))
    print("Norm of I - Q^TQ:\n {}".format(linalg.norm((np.identity(m) - Q1.dot(Q1.T)))))

    sQ, sR = slin.qr(A)
    print("Using pythons qr:\nNorm of I - Q^TQ:\n {}".format(linalg.norm((np.identity(m) - sQ.dot(sQ.T)))))


def problem5(A):
    m, n = A.shape
    OR = Orthogonalization(A.copy())
    Q, R = OR.givens_rotations()
    identity_norm = linalg.norm(np.identity(n) - Q.T.dot(Q))
    print("Norm of I - Q^TQ:\n{}".format(identity_norm))


def problem6():
    A = plt.imread('kvinna.jpg')
    U, S, VT = slin.svd(A, full_matrices=False)
    sigmamax = max(S)
    fig, ax_array = plt.subplots(1, 2)
    plt.subplots_adjust(bottom=0.2)

    ax_array[0].set_title('Compressed image')
    ax_array[1].set_title('Uncompressed image')
    ax_array[0].imshow(A)
    ax_array[1].imshow(A)

    slider_ax = plt.axes([0.3, 0.1, 0.4, 0.02])
    s = w.Slider(slider_ax, "Sigma max", 0, np.log10(sigmamax), valinit=np.log10(310))

    def compress(val):
        maximum_sigma = 10 ** s.val
        s.valtext.set_text(int(maximum_sigma))
        S_nu = np.diag(S)
        S_nu[S < maximum_sigma] = 0
        A_nu = U.dot(S_nu.dot(VT))
        ax_array[0].imshow(A_nu)

    s.on_changed(compress)
    plt.show()


m, n = 4, 4
A = np.random.randint(0, 1001, size=(m, n))
problem5(A)