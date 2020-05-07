import unittest
import numpy as np
from numpy import linalg as lin
from matricies import interpolator1d
from matricies import restrictor1d


class TestInterpolator(unittest.TestCase):
    def test_zero(self):
        # Tests that imputing the zero vector returns the zero vector in the new shape
        v = np.zeros(7).reshape(7,1)
        self.assertEqual(0, lin.norm(interpolator1d(v)))
        self.assertEqual(15, len(interpolator1d(v)))

        self.assertEqual(0, lin.norm(restrictor1d(v)))
        self.assertEqual(3, len(restrictor1d(v)))

    def test_ones(self):
        # test that imputing the constant vector outputs the same vector for the restrictor operator
        v = np.ones(15).reshape(15,1)
        self.assertEqual(np.sqrt(7), lin.norm(restrictor1d(v)))
        self.assertEqual(7, len(restrictor1d(v)))

        # testing that imputing the constant vector outputs a similar vector for the interpolator operator
        v = interpolator1d(v)
        print(v)
        self.assertEqual(31, len(v))
        self.assertEqual(np.sqrt(29.5), lin.norm(v))


