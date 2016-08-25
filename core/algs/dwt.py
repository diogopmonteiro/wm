from PIL import Image
from core.management.wm import WmError
import os
import numpy
from core.algs.abstract import Algorithm
from pywt import dwt2, idwt2
from core.algs.utils import TwoDimensionalDCT, Metrics


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
        return dwt2(r, self.WAVELET, mode='sym'), dwt2(g, self.WAVELET, mode='sym'), dwt2(b, self.WAVELET, mode='sym')

    def dwt_to_rgb(self, cr, cg, cb):
        return idwt2(cr, self.WAVELET, mode='sym'), idwt2(cg, self.WAVELET, mode='sym'), idwt2(cb, self.WAVELET, mode='sym')

    def rgb_to_dct(self, r, g, b):
        return TwoDimensionalDCT.forward(r), TwoDimensionalDCT.forward(g), TwoDimensionalDCT.forward(b)

    def dct_to_rgb(self, cr, cg, cb):
        return TwoDimensionalDCT.inverse(cr), TwoDimensionalDCT.inverse(cg), TwoDimensionalDCT.inverse(cb)

    def embed_specific(self, image, image_file, watermark=None):
        i_width = image.shape[0]
        i_height = image.shape[1]

        wm = watermark
        wm_width = wm.shape[0]
        wm_height = wm.shape[1]

        if not (wm_width == i_width / 2 and wm_height == i_height / 2):
            raise WmError("Watermark size must be half of image size. For now.")

        ir, ig, ib = self.split_image(image)

        img = self.split_image(wm)
        dcts_wm = self.rgb_to_dct(*img)

        dwts_i = self.rgb_to_dwt(ir, ig, ib)

        for color in range(3):
            dct = dcts_wm[color]
            hh = TwoDimensionalDCT.forward(dwts_i[color][1][2])
            for j in range(len(dct)):
                for k in range(len(dct[0])):
                    hh[j][k] += dct[j][k] * self.alpha

            ihh = TwoDimensionalDCT.inverse(hh)
            hh = dwts_i[color][1][2]
            for j in range(len(dct)):
                for k in range(len(dct[0])):
                    hh[j][k] = ihh[j][k]

        rgb = self.dwt_to_rgb(*dwts_i)

        img = self.join_image(*rgb)
        return img

    def extract_specific(self, image, watermark):
        wm = self.open_image(watermark)

        RED, GREEN, BLUE = (0,1,2)

        image_rgb = self.split_image(image)
        wm_rgb = self.split_image(wm)

        image_dwt = list(self.rgb_to_dwt(*image_rgb))
        wm_dwt = list(self.rgb_to_dwt(*wm_rgb))

        for color in range(3):
            dct_image_dwt = TwoDimensionalDCT.forward(image_dwt[color][1][2])
            dct_wm_dwt = TwoDimensionalDCT.forward(wm_dwt[color][1][2])
            for i in range(len(image_dwt[RED][1][2])):
                for j in range(len(image_dwt[RED][1][2][0])):
                    dct_image_dwt[i][j] = (dct_wm_dwt[i][j] - dct_image_dwt[i][j])/self.alpha

            for i in range(len(image_dwt[RED][1][2])):
                for j in range(len(image_dwt[RED][1][2][0])):
                    image_dwt[color][1][2][i][j] = dct_image_dwt[i][j]

        return self.join_image(*self.dct_to_rgb(image_dwt[RED][1][2], image_dwt[GREEN][1][2], image_dwt[BLUE][1][2])), 0

    def get_watermark_name(self, filename):
        _, filename = os.path.split(filename)
        return self.get_image_output_file(_+"watermark-"+filename.split('.')[0]+".png")


    def benchmark_extract_step(self, path, image, attack_name, attacked, attacked_path, watermark_file):
        wmark, gamma = self.extract_specific(numpy.array(Image.open(image)),
                                                          attacked_path)
        wmark = wmark.clip(0,255)
        wmark = wmark.astype('uint8')
        gamma = Metrics.gamma(wmark.ravel(), numpy.array(Image.open(watermark_file)).ravel())
        wmark = Image.fromarray(wmark)
        wmark.save(os.path.join(path, attack_name + "-extracted-wm-" + os.path.split(image)[1]))
        return wmark, gamma






