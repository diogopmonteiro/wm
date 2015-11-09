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

        k = len(f_r)-1
        y = len(f_r[0])-1
        for i in range(len(cD1r)):
            for j in range(len(cD1r)):
                cD1r[i][j] += f_r[i][j]
                cD1b[i][j] += f_b[i][j]
                cD1g[i][j] += f_g[i][j]
                if j >= y:
                    break
            if i >= k:
                break

        coeffs_r = cA3r, (cH3r, cV3r, cD3r), (cH2r, cV2r, cD2r), (cH1r, cV1r, cD1r)
        coeffs_b = cA3b, (cH3b, cV3b, cD3b), (cH2b, cV2b, cD2b), (cH1b, cV1b, cD1b)
        coeffs_g = cA3g, (cH3g, cV3g, cD3g), (cH2g, cV2g, cD2g), (cH1g, cV1g, cD1g)



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

        watermark = self.open_image(watermark)

        wr = []
        wb = []
        wg = []
        for row in watermark:
            wr_col = []
            wb_col = []
            wg_col = []
            for col in row:
                wr_col.append(col[0])
                wb_col.append(col[1])
                wg_col.append(col[2])
            wr.append(wr_col)
            wb.append(wb_col)
            wg.append(wg_col)

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

        coeffs_wr = wavedec2(wr, self.WAVELET, level=3)
        cA3r, (cH3r, cV3r, cD3r), (cH2r, cV2r, cD2r), (cH1r, cV1r, cD1wr) = coeffs_wr

        coeffs_wb = wavedec2(wb, self.WAVELET, level=3)
        cA3b, (cH3b, cV3b, cD3b), (cH2b, cV2b, cD2b), (cH1b, cV1b, cD1wb) = coeffs_wb

        coeffs_wg = wavedec2(wg, self.WAVELET, level=3)
        cA3g, (cH3g, cV3g, cD3g), (cH2g, cV2g, cD2g), (cH1g, cV1g, cD1wg) = coeffs_wg

        img = []
        i = 0
        j = 0
        line = []

        for row in range(len(cD1b)):
            for col in range(len(cD1b[0])):
                line.append([cD1r[row][col]-cD1wr[row][col],
                             cD1b[row][col]-cD1wb[row][col],
                             cD1g[row][col]-cD1wg[row][col]])
                i+=1
                if i == 100:
                    i = 0
                    img.append(line)
                    line = []
                    j += 1
                    if j == 100:
                        break
            if j == 100:
                break

        inverse = TwoDimensionalDCT.inverse(img)
        return inverse