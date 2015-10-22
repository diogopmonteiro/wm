from scipy.fftpack import dct, idct


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
