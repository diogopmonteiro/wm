import scipy
from scipy.stats import stats
from scipy.fftpack import dct, idct
import math


class TwoDimensionalDCT(object):
    """
        Utility class that calculates a two dimensional DCT for a specified image
        http://stackoverflow.com/questions/15978468/using-the-scipy-dct-function-to-create-a-2d-dct-ii
    """

    @classmethod
    def forward(cls, image):
        return dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')

    @classmethod
    def inverse(cls, dct_image):
        return idct(idct(dct_image, axis=0, norm='ortho'), axis=1, norm='ortho')


class Metrics(object):

    @classmethod
    def mse(cls, i, iw):
        """
        Calculates the mean square error between two images
        :param i: the original image
        :param iw: the watermarked image
        :return: the mean square error value
        """
        return ((i - iw) ** 2).mean(axis=None)

    @classmethod
    def psnr(cls, i, iw):
        """
        The Peak Signal to Noise Ratio is used to measure the imperceptibility of the
        watermarked and the extracted watermark images
        :param i: the original image
        :param iw: the watermarked image
        :return: the value of Peak Signal to Noise Ratio
        """
        imax = i.max()
        iwmax = iw.max()
        m = imax if imax > iwmax else iwmax
        mse = cls.mse(i,iw)
        if mse == 0.0:
            return 100.0

        return 10 * math.log10(pow(m,2)/math.sqrt(cls.mse(i,iw)))

        """@classmethod
        def gamma(cls, w, o):

        top = 0
        bw = 0
        bo = 0
        for i in range(len(w)):
            top += w[i] * o[i]
            bw += w[i] * w[i]
            bo += o[i] * o[i]
        bottom = math.sqrt(bw)*math.sqrt(bo)
        if bottom == 0:
            return 1
        return abs(top/bottom)"""

    @classmethod
    def gamma(cls, w, o):
        return stats.pearsonr(w,o)[0]
