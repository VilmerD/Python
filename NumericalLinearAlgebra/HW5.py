from scipy.linalg import *
from matplotlib import pyplot as plt
from matplotlib.widgets import *


def shift_A(A0, p):
    if p == 0:
        return np.diag(np.diag(A0))
    elif p == 1:
        return A0
    d = np.diag(A0)
    scaled = A0*p
    return scaled + (1 - p)*np.diag(d)


def App():
    A0 = np.array([[5, 0, 0, -1], [1, 0, -1, 1], [-1.5, 1, -2, 1], [-1, 1, 3, -3]])
    w, _ = eig(A0)
    wreal = w.real
    fig, ax = plt.subplots()
    plt.plot([-10, 10], [0, 0], 'k')
    plt.subplots_adjust(bottom=0.3)
    slider_ax = plt.axes([0.3, 0.1, 0.4, 0.1])
    s = Slider(slider_ax, "Shift, p", 0, 1, valinit=0.5)

    marker_height = 0.3
    line = [marker_height / 2, -marker_height / 2]
    dx_D = [-0.4, -0.3, -0.15, -0.15]
    dx_lambda = [0, -0.2, -0.1, -0.1]
    for k in range(1, 5):
        d = A0[k-1, k-1]
        ax.plot([d, d], line, 'k')
        ax.text(d + dx_D[k-1], -0.7, "$D_{}$".format(k))

        wrealk = wreal[k-1]
        ax.plot([wrealk, wrealk], line, 'k')
        ax.text(wrealk + dx_lambda[k-1], -0.7, '$\lambda$')

    circles = []
    eigs, radii = eigs_and_radii(A0, 100)
    peigs = ax.plot(eigs[:, 50].real, [0] * 4, 'yo', label='Eigenvalues')

    for k in range(0, 4):
        ax.add_artist(plt.Circle((A0[k, k], 0), radii[k, -1], fill=False, alpha=0.1, edgecolor='k'))
        circles.append(plt.Circle((A0[k, k], 0), radii[k, 50], alpha=0.3, edgecolor='k'))
        ax.add_artist(circles[k])

    def update_rings(val):
        update_rings.k = update_rings.k+1
        if update_rings.k % 10 == 0:
            plt.pause(0.03)
            # Super strange bug with an even stranger solution...
            # The issue is somehow related to pyplot using multiple threads.
        p = int((s.val * 100))
        ei = eigs[:, p]
        di = radii[:, p]
        for k in range(0, 4):
            dik = di[k]
            circles[k].set_radius(dik)
        peigs[0].set_data(ei.real, [0] * 4)

    update_rings.k = 0

    s.on_changed(update_rings)
    ax.axis([-8, 8, -6, 6])
    ax.legend()
    ax.set_ylabel('Imaginary axis')
    ax.set_xlabel('Real axis')
    plt.show()


def eigs_and_radii(A0, samples):
    eigs = np.zeros((4, samples + 1))
    radii = np.zeros((4, samples + 1))
    for k in range(0, samples + 1):
        Ap = shift_A(A0, k / samples)
        w, v = eig(Ap)
        eigs[:, k] = w.real
        for j in range(0, 4):
            radii[j, k] = sum(abs(Ap[j, :])) - abs(Ap[j, j])
    return eigs, radii


def QR_iteration(A):
    if A.shape[0] == 1 or A.shape[1] == 1:
        return A, 0
    m = int(A.shape[0] / 2)
    k = 1
    tol = 1e-8
    Q, _ = qr(A)
    A = np.dot(Q, np.dot(A, Q.T))
    while True:
        u = A[m, m] * np.eye(A.shape[0])
        Q, R = qr(A - u)
        A = np.dot(R, Q) + u
        upper = abs(np.diag(A, k=1))
        zeros = np.where(upper < tol)[0]
        if len(zeros) > 0:
            row = zeros[0]
            A1 = A[:row + 1, :row + 1]
            A2 = A[row + 1:, row + 1:]
            A1, k1 = QR_iteration(A1)
            A2, k2 = QR_iteration(A2)
            return block_diag(A1, A2), k1 + k2 + k
        k += 1


def poly_generator(A):
    s = A.shape[0]
    a = np.diag(A)
    b = np.diag(A, k=1)

    def p(x):
        dets = np.zeros((s + 2, 1))
        dets[1] = 1
        dets[2] = (a[0] - x)
        signs = 0
        if dets[2] < 0:
            signs = 1
        for k in range(3, s + 2):
            d = (a[k - 2] - x)*dets[k-1] - b[k - 3]**2*dets[k-2]
            dets[k] = d
            if d < 0 and dets[k - 1] >= 0 or d > 0 and dets[k - 1] <= 0:
                signs += 1
        return signs
    return p


def Gerschgorin_interval(A):
    high, low = 0, 0
    for r in range(0, A.shape[0]):
        d = sum(abs(A[r, :])) - abs(A[r, r])
        high = max(high, A[r, r] + d)
        low = min(low, A[r, r] - d)
    return high, low


def Bisection(A, tol=1e-8):
    poly = poly_generator(A)
    high, low = Gerschgorin_interval(A)
    eigs = []

    def Bisection_inner(n_eigs_left, n_eigs_right, left_coord, right_coord):
        m = (left_coord + right_coord) / 2
        if abs((right_coord - left_coord) / 2) < tol:
            eigs.append(m)
        else:
            n_eigs_mid = poly(m)
            n_eigs_left_half = n_eigs_mid - n_eigs_left
            n_eigs_right_half = n_eigs_right - n_eigs_mid
            if n_eigs_left_half > 0:
                Bisection_inner(n_eigs_left, n_eigs_mid, left_coord, m)
            if n_eigs_right_half > 0:
                Bisection_inner(n_eigs_mid, n_eigs_right, m, right_coord)

    Bisection_inner(poly(low), poly(high), low, high)
    return np.array(eigs)


def task2():
    s = 15
    for n in [10, 100, 1000]:
        tnits = 0
        for m in range(0, 1000):
            A = np.random.rand(s, s)
            A = (A + A.T) / 2
            _, nits = QR_iteration(A)
            tnits += nits
        print("Avg num_its with n={}: {}".format(n, tnits/m))


def task4():
    n = 8
    A = np.random.rand(8, 8)
    A = (A + A.T) / 2
    A, _ = hessenberg(A)
    print("Matrix: \n{}".format(A))
    print("Eigenvalues with eig: \n{}".format(np.sort(eig(A)[0])))
    print("Eigenvalues with bisection: \n{}".format(Bisection(A)))


if __name__ == '__main__':
    App()
