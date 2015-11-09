import numpy
from core.algs.abstract import Algorithm
from pywt import dwt2, idwt2
from core.algs.utils import TwoDimensionalDCT


class DWT(Algorithm):

    WAVELET = 'haar'
    alpha = 0.1

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

    def embed_specific(self, image, image_file, watermark=None):
        wm = watermark
        ir, ig, ib = self.split_image(image)

        dct_wm = TwoDimensionalDCT.forward(wm)
        dcts_wm = self.split_image(dct_wm)

        dwts_i = self.rgb_to_dwt(ir, ig, ib)

        for i in range(3):
            dct = dcts_wm[i]
            ll = dwts_i[i][0]
            for j in range(len(dct)):
                for k in range(len(dct[0])):
                    ll[j][k] += dct[j][k] * self.alpha

        rgb = self.dwt_to_rgb(*dwts_i)
        return self.join_image(*rgb)

    def extract_specific(self, image, watermark):
        wm = self.open_image(watermark)

        RED, GREEN, BLUE = (0,1,2)
        LL, LH, HL, HH = (0, (1,0), (1,1), (1,2))

        W = LL

        image_rgb = self.split_image(image)
        wm_rgb = self.split_image(wm)

        image_dwt = list(self.rgb_to_dwt(*image_rgb))
        wm_dwt = list(self.rgb_to_dwt(*wm_rgb))

        for color in range(3):
            for i in range(len(image_dwt[RED][W])):
                for j in range(len(image_dwt[RED][W][0])):
                    image_dwt[color][W][i][j] = (wm_dwt[color][W][i][j] - image_dwt[color][W][i][j])/self.alpha

        dct_r = TwoDimensionalDCT.inverse(image_dwt[RED][W])
        dct_g = TwoDimensionalDCT.inverse(image_dwt[GREEN][W])
        dct_b = TwoDimensionalDCT.inverse(image_dwt[BLUE][W])

        return self.join_image(dct_r,dct_g, dct_b)


    def get_watermark_name(self, filename):
        pass






