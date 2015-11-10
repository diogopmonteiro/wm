import os
import numpy
from core.algs.abstract import Algorithm
from pywt import dwt2, idwt2
from core.algs.utils import TwoDimensionalDCT, Metrics


class DWT(Algorithm):

    WAVELET = 'db1'
    alpha = 0.01

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

    def dwt_to_rgb(self, cr, cg, cb):
        return idwt2(cr, self.WAVELET), idwt2(cg, self.WAVELET), idwt2(cb, self.WAVELET)

    def rgb_to_dct(self, r, g, b):
        return TwoDimensionalDCT.forward(r), TwoDimensionalDCT.forward(g), TwoDimensionalDCT.forward(b)

    def dct_to_rgb(self, cr, cg, cb):
        return TwoDimensionalDCT.inverse(cr), TwoDimensionalDCT.inverse(cg), TwoDimensionalDCT.inverse(cb)

    def embed_specific(self, image, image_file, watermark=None):
        wm = watermark
        ir, ig, ib = self.split_image(image)

        img = self.split_image(wm)
        dcts_wm = self.rgb_to_dct(*img)

        xr, xg, xb = self.dct_to_rgb(*dcts_wm)

        dwts_i = self.rgb_to_dwt(ir, ig, ib)

        for color in range(3):
            dct = dcts_wm[color]
            hh = dwts_i[color][1][2]
            for j in range(len(dct)):
                for k in range(len(dct[0])):
                    hh[j][k] += dct[j][k] * self.alpha

        rgb = self.dwt_to_rgb(*dwts_i)
        img = self.join_image(*rgb)
        print img
        return img

    def extract_specific(self, image, watermark):
        wm = self.open_image(watermark)

        RED, GREEN, BLUE = (0,1,2)
        LL, LH, HL, HH = (0, (1,0), (1,1), (1,2))

        W = HH

        image_rgb = self.split_image(image)
        wm_rgb = self.split_image(wm)

        image_dwt = list(self.rgb_to_dwt(*image_rgb))
        wm_dwt = list(self.rgb_to_dwt(*wm_rgb))

        for color in range(3):
            for i in range(len(image_dwt[RED][1][2])):
                for j in range(len(image_dwt[RED][1][2][0])):
                    image_dwt[color][1][2][i][j] = (wm_dwt[color][1][2][i][j] - image_dwt[color][1][2][i][j])/self.alpha

        return self.join_image(*self.dct_to_rgb(image_dwt[RED][1][2], image_dwt[GREEN][1][2], image_dwt[BLUE][1][2])), 0

    def get_watermark_name(self, filename):
        _, filename = os.path.split(filename)
        return self.get_image_output_file(_+"watermark-"+filename.split('.')[0]+".png")






