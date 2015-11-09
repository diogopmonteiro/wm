import numpy
from core.algs.abstract import Algorithm
<<<<<<< 11eb72185b1839bef1e8abfd260e4dc05a18131a
from pywt import dwt2
=======
from pywt import idwt2
>>>>>>> dwt_to_rgb


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

    def dwt_ro_rgb(self, cr, cg, cb):
        return idwt2(cr, self.WAVELET), idwt2(cg, self.WAVELET), idwt2(cb, self.WAVELET)

    def embed_specific(self, image, image_file, watermark=None):
        pass

    def extract_specific(self, image, watermark):
        pass

