from unittest import TestCase

import numpy
from core.algs.utils import TwoDimensionalDCT

'''
Tests derived from http://www.whydomath.org/node/wavlets/dct.html
'''

class TwoDimensionalDCTTest(TestCase):

    def closeEnough(self, x, y):
        return abs(x-y) < 2

    def setUp(self):
        self.close_values = numpy.array([
            [[51, 51, 51], [52, 52, 52], [51, 51, 51], [50, 50, 50], [50, 50, 50], [52, 52, 52], [50, 50, 50], [52, 52, 52]],
            [[51, 51, 51], [52, 52, 52], [51, 51, 51], [51, 51, 51], [50, 50, 50], [52, 52, 52], [52, 52, 52], [51, 51, 51]],
            [[50, 50, 50], [50, 50, 50], [51, 51, 51], [52, 52, 52], [52, 52, 52], [51, 51, 51], [51, 51, 51], [51, 51, 51]],
            [[51, 51, 51], [50, 50, 50], [50, 50, 50], [50, 50, 50], [52, 52, 52], [50, 50, 50], [50, 50, 50], [51, 51, 51]],
            [[51, 51, 51], [50, 50, 50], [50, 50, 50], [51, 51, 51], [50, 50, 50], [50, 50, 50], [51, 51, 51], [50, 50, 50]],
            [[50, 50, 50], [51, 51, 51], [52, 52, 52], [52, 52, 52], [51, 51, 51], [50, 50, 50], [50, 50, 50], [50, 50, 50]],
            [[51, 51, 51], [52, 52, 52], [51, 51, 51], [50, 50, 50], [52, 52, 52], [50, 50, 50], [52, 52, 52], [50, 50, 50]],
            [[50, 50, 50], [51, 51, 51], [52, 52, 52], [52, 52, 52], [50, 50, 50], [51, 51, 51], [52, 52, 52], [51, 51, 51]]
        ])

        self.solid_gray = numpy.array([
            [[100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100]],
            [[100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100]],
            [[100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100]],
            [[100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100]],
            [[100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100]],
            [[100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100]],
            [[100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100]],
            [[100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100]]
        ])

    def test_dct_solid_gray(self):
        dct = TwoDimensionalDCT.forward(self.solid_gray)
        for channel in range(3):
            self.assertTrue(self.closeEnough(dct[0][0][channel], 800))

        for channel in range(3):
            for i in range(len(dct)):
                for j in range(len(dct[i])):
                    if i != 0 and j != 0:
                        self.assertTrue(self.closeEnough(dct[i][j][channel], 0))

    def test_dct_close_values(self):
        dct = TwoDimensionalDCT.forward(self.close_values)
        self.assertTrue(self.closeEnough(dct[0][0][0], 407))

    def _test_inverse(self, img, original):
        for channel in range(3):
            for i in range(len(img)):
                for j in range(len(img[i])):
                    self.assertTrue(self.closeEnough(img[i][j][channel], original[i][j][channel]))

    def test_idct_solid_gray(self):
        img = TwoDimensionalDCT.inverse(TwoDimensionalDCT.forward(self.solid_gray))
        self._test_inverse(img, self.solid_gray)

    def test_idct_close_values(self):
        img = TwoDimensionalDCT.inverse(TwoDimensionalDCT.forward(self.close_values))
        self._test_inverse(img, self.close_values)