import unittest
import numpy as np
from numpy.linalg import norm
from numpy import linalg as lin
from Multigrid.matricies import interpolator2d
from Multigrid.matricies import restrictor2d


class TestInterpolator(unittest.TestCase):
    def test_zero(self):
        n = 7
        v = np.zeros(n ** 2)
        q = interpolator2d(v)
        self.assertEqual(0, norm(q), "Normen ska vara noll")

    def test_ones(self):
        n = 3
        v = np.ones(n ** 2)
        q = interpolator2d(v)

        print(q.reshape((7, 7)))


class TestRestrictor(unittest.TestCase):
    def test_zero(self):
        n = 3
        v = np.zeros(n ** 2)
        q = restrictor2d(v)
        self.assertEqual(0, norm(q), "Normen ska vara noll")

    def test_ones(self):
        n = 7
        ns = int((n - 1) / 2)
        v = np.ones(n ** 2)
        q = restrictor2d(v)

        print(q.reshape((ns, ns)))
