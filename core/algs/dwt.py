from abstract import Algorithm
from core.algs.utils import TwoDimensionalDCT
from scipy.misc import imread
from pywt import wavedec2, waverec2
import numpy


class DWT(Algorithm):

    WAVELET = 'db1'

    def embed_specific(self, image, image_file, watermark=None):
        f_dct = TwoDimensionalDCT.forward(watermark)
        f_r = []
        f_b = []
        f_g = []

        for row in f_dct:
            f_r_col = []
            f_b_col = []
            f_g_col = []
            for col in row:
                f_r_col.append(col[0])
                f_b_col.append(col[1])
                f_g_col.append(col[2])
            f_r.append(f_r_col)
            f_b.append(f_b_col)
            f_g.append(f_g_col)


        r = []
        b = []
        g = []
        for row in image:
            r_col = []
            b_col = []
            g_col = []
            for col in row:
                r_col.append(col[0])
                b_col.append(col[1])
                g_col.append(col[2])
            r.append(r_col)
            b.append(b_col)
            g.append(g_col)

        coeffs_r = wavedec2(r, self.WAVELET, level=3)
        cA3r, (cH3r, cV3r, cD3r), (cH2r, cV2r, cD2r), (cH1r, cV1r, cD1r) = coeffs_r

        coeffs_b = wavedec2(b, self.WAVELET, level=3)
        cA3b, (cH3b, cV3b, cD3b), (cH2b, cV2b, cD2b), (cH1b, cV1b, cD1b) = coeffs_b

        coeffs_g = wavedec2(g, self.WAVELET, level=3)
        cA3g, (cH3g, cV3g, cD3g), (cH2g, cV2g, cD2g), (cH1g, cV1g, cD1g) = coeffs_g


        for i in range(len(f_r)):
            cD1r[i] += f_r_col[i]
            cD1b[i] += f_b_col[i]
            cD1g[i] += f_g_col[i]

        r = waverec2(coeffs_r, self.WAVELET)
        b = waverec2(coeffs_b, self.WAVELET)
        g = waverec2(coeffs_g, self.WAVELET)

        img = []

        for row in range(len(r)):
            line = []
            for col in range(len(r[0])):
                line.append([r[row][col], b[row][col], g[row][col]])
            img.append(line)
        return numpy.array(img)


    def extract_specific(self, image, watermark):
        pass