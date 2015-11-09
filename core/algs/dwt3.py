import numpy
from core.algs.abstract import Algorithm
from pywt import dwt2


class DWT(Algorithm):

    WAVELET = 'haar'

    def split_image(self, image):
        return image[:, :, 2], image[:, :, 1], image[:, :, 0]

    def join_image(self, r, g, b):
        result = numpy.zeros(r.shape + (3,))
        color = 0
        for cm in [b,g,r]:
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    result[i][j][color] = cm[i][j]
            color += 1
        return result

    def rgb_to_dwt(self, r, g, b):
        return dwt2(r, self.WAVELET), dwt2(g, self.WAVELET), dwt2(b, self.WAVELET)

    def embed_specific(self, image, image_file, watermark=None):
        pass

    def extract_specific(self, image, watermark):
        pass
