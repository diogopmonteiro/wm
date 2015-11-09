from abstract import Algorithm
from core.algs.utils import TwoDimensionalDCT
from scipy.misc import imread
from pywt import dwt2, idwt2
import numpy
from PIL import Image
import json
from core.settings import PROJECT_CODE_DIRECTORY
import os

class DWT(Algorithm):

    WAVELET = 'db1'
    N = 16
    alpha = .60

    def embed_specific(self, image, image_file, watermark=None):
        r = image[:,:,2]
        g = image[:,:,1]
        b = image[:,:,0]

        #r, g, b = image.T
        r_matrix = {}
        g_matrix = {}
        b_matrix = {}
        k = len(r)/self.N
        z = len(r[0])/self.N
        x = 0
        y = 0
        for i in range(k, len(r)+1, k):
            for j in range(z, len(r[0])+1, z):
                r_matrix[(x, y)] = r[i-k:i, j-z:j]
                g_matrix[(x, y)] = g[i-k:i, j-z:j]
                b_matrix[(x, y)] = b[i-k:i, j-z:j]
                y += 1
            x += 1
            y = 0
        zz = self.zigzag(self.N)
        r_matrix_n = {}
        g_matrix_n = {}
        b_matrix_n = {}

        for key in r_matrix:
            r_matrix_n[key] = r_matrix[self.index_tuple(zz[key], self.N)]
        for key in r_matrix:
            g_matrix_n[key] = g_matrix[self.index_tuple(zz[key], self.N)]
        for key in r_matrix:
            b_matrix_n[key] = b_matrix[self.index_tuple(zz[key], self.N)]

        r_matrix_nn = None
        g_matrix_nn = None
        b_matrix_nn = None

        for lin in range(self.N):
            r_m = r_matrix_n[(lin, 0)]
            g_m = g_matrix_n[(lin, 0)]
            b_m = b_matrix_n[(lin, 0)]
            for col in range(1, self.N):
                r_m = numpy.concatenate((r_m, r_matrix_n[(lin, col)]), axis=1)
                g_m = numpy.concatenate((g_m, g_matrix_n[(lin, col)]), axis=1)
                b_m = numpy.concatenate((b_m, b_matrix_n[(lin, col)]), axis=1)
            if r_matrix_nn is None:
                r_matrix_nn = r_m
                g_matrix_nn = g_m
                b_matrix_nn = b_m
            else:
                r_matrix_nn = numpy.concatenate((r_matrix_nn, r_m), axis=0)
                g_matrix_nn = numpy.concatenate((g_matrix_nn, g_m), axis=0)
                b_matrix_nn = numpy.concatenate((b_matrix_nn, b_m), axis=0)


        coeffs_r = dwt2(r_matrix_nn, self.WAVELET)
        cA1r, (cH1r, cV1r, cD1r) = coeffs_r

        SH1_r = TwoDimensionalDCT.forward(cH1r)
        SH2_r = TwoDimensionalDCT.forward(cV1r)
        SH3_r = TwoDimensionalDCT.forward(cD1r)

        coeffs_g = dwt2(g_matrix_nn, self.WAVELET)
        cA1g, (cH1g, cV1g, cD1g) = coeffs_g

        SH1_g = TwoDimensionalDCT.forward(cH1g)
        SH2_g = TwoDimensionalDCT.forward(cV1g)
        SH3_g = TwoDimensionalDCT.forward(cD1g)


        coeffs_b = dwt2(b_matrix_nn, self.WAVELET)
        cA1b, (cH1b, cV1b, cD1b) = coeffs_b

        SH1_b = TwoDimensionalDCT.forward(cH1b)
        SH2_b = TwoDimensionalDCT.forward(cV1b)
        SH3_b = TwoDimensionalDCT.forward(cD1b)


        rw = watermark[:,:,2]
        gw = watermark[:,:,1]
        bw = watermark[:,:,0]

        coeffs_rw = dwt2(rw, self.WAVELET)
        cA1rw, (cH1rw, cV1rw, cD1rw) = coeffs_rw

        SH1_rw = TwoDimensionalDCT.forward(cH1rw)
        SH2_rw = TwoDimensionalDCT.forward(cV1rw)
        SH3_rw = TwoDimensionalDCT.forward(cD1rw)


        coeffs_gw = dwt2(gw, self.WAVELET)
        cA1gw, (cH1gw, cV1gw, cD1gw) = coeffs_gw

        SH1_gw = TwoDimensionalDCT.forward(cH1gw)
        SH2_gw = TwoDimensionalDCT.forward(cV1gw)
        SH3_gw = TwoDimensionalDCT.forward(cD1gw)


        coeffs_bw = dwt2(bw, self.WAVELET)
        cA1bw, (cH1bw, cV1bw, cD1bw) = coeffs_bw

        SH1_bw = TwoDimensionalDCT.forward(cH1bw)
        SH2_bw = TwoDimensionalDCT.forward(cV1bw)
        SH3_bw = TwoDimensionalDCT.forward(cD1bw)

        print(1,SH1_r)

        for i in range(len(SH1_bw)):
            for j in range(len(SH1_bw[0])):
                SH1_r[i][j] += SH1_rw[i][j]*self.alpha
                SH1_g[i][j] += SH1_gw[i][j]*self.alpha
                SH1_b[i][j] += SH1_bw[i][j]*self.alpha
                SH2_r[i][j] += SH2_rw[i][j]*self.alpha
                SH2_g[i][j] += SH2_gw[i][j]*self.alpha
                SH2_b[i][j] += SH2_bw[i][j]*self.alpha
                SH3_r[i][j] += SH3_rw[i][j]*self.alpha
                SH3_g[i][j] += SH3_gw[i][j]*self.alpha
                SH3_b[i][j] += SH3_bw[i][j]*self.alpha
        print(SH1_rw)
        print(2,SH1_r)


        iR1 = TwoDimensionalDCT.inverse(SH1_r)
        iR2 = TwoDimensionalDCT.inverse(SH2_r)
        iR3 = TwoDimensionalDCT.inverse(SH3_r)

        print(3,TwoDimensionalDCT.forward(iR1))

        coeffs_r = cA1r, (iR1, iR2, iR3)

        r = idwt2(coeffs_r, self.WAVELET)

        coeffs_r = dwt2(r, self.WAVELET)
        cA1r, (cH1r, cV1r, cD1r) = coeffs_r

        print(4,TwoDimensionalDCT.forward(cH1r))


        iG1 = TwoDimensionalDCT.inverse(SH1_g)
        iG2 = TwoDimensionalDCT.inverse(SH2_g)
        iG3 = TwoDimensionalDCT.inverse(SH3_g)

        coeffs_g = cA1g, (iG1, iG2, iG3)

        g = idwt2(coeffs_g, self.WAVELET)

        iB1 = TwoDimensionalDCT.inverse(SH1_b)
        iB2 = TwoDimensionalDCT.inverse(SH2_b)
        iB3 = TwoDimensionalDCT.inverse(SH3_b)

        coeffs_b = cA1b, (iB1, iB2, iB3)

        b = idwt2(coeffs_b, self.WAVELET)


        dd = {}
        for key in zz:
            x = key[0]
            y = key[1]
            dd[self.index_tuple(zz[key], self.N)] = (x*self.N)+y

        r_matrix = {}
        g_matrix = {}
        b_matrix = {}
        k = len(r)/self.N
        z = len(r[0])/self.N
        x = 0
        y = 0
        for i in range(k, len(r)+1,k):
            for j in range(z, len(r[0])+1, z):
                r_matrix[(x, y)] = r[i-k:i, j-z:j]
                g_matrix[(x, y)] = g[i-k:i, j-z:j]
                b_matrix[(x, y)] = b[i-k:i, j-z:j]
                y += 1
            x += 1
            y = 0

        rf = {}
        gf = {}
        bf = {}
        for key in r_matrix:
            rf[key] = r_matrix[self.index_tuple(dd[key], self.N)]
        for key in g_matrix:
            gf[key] = g_matrix[self.index_tuple(dd[key], self.N)]
        for key in b_matrix:
            bf[key] = b_matrix[self.index_tuple(dd[key], self.N)]

        r_matrix_n = rf
        g_matrix_n = gf
        b_matrix_n = bf

        r_matrix_nn = None
        g_matrix_nn = None
        b_matrix_nn = None

        for lin in range(self.N):
            r_m = r_matrix_n[(lin, 0)]
            g_m = g_matrix_n[(lin, 0)]
            b_m = b_matrix_n[(lin, 0)]
            for col in range(1, self.N):
                r_m = numpy.concatenate((r_m, r_matrix_n[(lin, col)]), axis=1)
                g_m = numpy.concatenate((g_m, g_matrix_n[(lin, col)]), axis=1)
                b_m = numpy.concatenate((b_m, b_matrix_n[(lin, col)]), axis=1)
            if r_matrix_nn is None:
                r_matrix_nn = r_m
                g_matrix_nn = g_m
                b_matrix_nn = b_m
            else:
                r_matrix_nn = numpy.concatenate((r_matrix_nn, r_m), axis=0)
                g_matrix_nn = numpy.concatenate((g_matrix_nn, g_m), axis=0)
                b_matrix_nn = numpy.concatenate((b_matrix_nn, b_m), axis=0)

        final = []
        for lin in range(len(r_matrix_nn)):
            line = []
            for col in range(len(r_matrix_nn[0])):
                line.append([b_matrix_nn[lin][col], g_matrix_nn[lin][col], r_matrix_nn[lin][col]])

            final.append(line)
        self.export_image(numpy.array(final),image_file)
        print(self.start_op(numpy.array(final))[0][0])
        return numpy.array(final)

    def extract_specific(self, image, watermark):

        image = self.load_watermark(image)

        wm = self.open_image(watermark)
        colors = self.start_op(image)

        colorsO = self.start_op(wm)

        result = numpy.empty([3,3,len(colors[0][0]), len(colors[0][0][0])])

        for color in range(3):
            for i in range(len(colors[color][0])):
                for j in range(len(colors[color][0][0])):
                    result[color][0][i][j] = (colors[color][0][i][j] - colorsO[color][0][i][j]) / self.alpha
                    result[color][1][i][j] = (colors[color][1][i][j] - colorsO[color][1][i][j]) / self.alpha
                    result[color][2][i][j] = (colors[color][2][i][j] - colorsO[color][2][i][j]) / self.alpha


        iR1 = TwoDimensionalDCT.inverse(result[0][0])
        iR2 = TwoDimensionalDCT.inverse(result[0][1])
        iR3 = TwoDimensionalDCT.inverse(result[0][2])

        coeffs_r = colors[0][3], (iR1, iR2, iR3)

        r = idwt2(coeffs_r, self.WAVELET)


        iG1 = TwoDimensionalDCT.inverse(result[1][0])
        iG2 = TwoDimensionalDCT.inverse(result[1][1])
        iG3 = TwoDimensionalDCT.inverse(result[1][2])

        coeffs_g = colors[1][3], (iG1, iG2, iG3)

        g = idwt2(coeffs_g, self.WAVELET)

        iB1 = TwoDimensionalDCT.inverse(result[2][0])
        iB2 = TwoDimensionalDCT.inverse(result[2][1])
        iB3 = TwoDimensionalDCT.inverse(result[2][2])

        coeffs_b = colors[2][3], (iB1, iB2, iB3)

        b = idwt2(coeffs_b, self.WAVELET)

        final = []
        for lin in range(len(r)):
            line = []
            for col in range(len(r[0])):
                line.append([b[lin][col], g[lin][col], r[lin][col]])
            final.append(line)

        return numpy.array(final), None

    def zigzag(self, n):
        index_order = sorted(((x, y) for x in range(n) for y in range(n)), key=lambda (x, y): (x+y, -y if (x+y) % 2 else y))
        return {index: n for n, index in enumerate(index_order)}

    def index_tuple(self, i, n):
        for lin in range(n):
            for col in range(n):
                if i == 0:
                    return lin, col
                i -= 1

    def printzz(self, my_array):
        n = int(len(my_array) ** 0.5 + 0.5)
        for x in range(n):
            for y in range(n):
                    print "%2i" % my_array[(x, y)],
            print

    def start_op(self, image):
        r = image[:,:,2]
        g = image[:,:,1]
        b = image[:,:,0]

        #r, g, b = image.T
        r_matrix = {}
        g_matrix = {}
        b_matrix = {}
        k = len(r)/self.N
        z = len(r[0])/self.N
        x = 0
        y = 0
        for i in range(k, len(r)+1, k):
            for j in range(z, len(r[0])+1, z):
                r_matrix[(x, y)] = r[i-k:i, j-z:j]
                g_matrix[(x, y)] = g[i-k:i, j-z:j]
                b_matrix[(x, y)] = b[i-k:i, j-z:j]
                y += 1
            x += 1
            y = 0
        zz = self.zigzag(self.N)
        r_matrix_n = {}
        g_matrix_n = {}
        b_matrix_n = {}

        for key in r_matrix:
            r_matrix_n[key] = r_matrix[self.index_tuple(zz[key], self.N)]
        for key in r_matrix:
            g_matrix_n[key] = g_matrix[self.index_tuple(zz[key], self.N)]
        for key in r_matrix:
            b_matrix_n[key] = b_matrix[self.index_tuple(zz[key], self.N)]

        r_matrix_nn = None
        g_matrix_nn = None
        b_matrix_nn = None

        for lin in range(self.N):
            r_m = r_matrix_n[(lin, 0)]
            g_m = g_matrix_n[(lin, 0)]
            b_m = b_matrix_n[(lin, 0)]
            for col in range(1, self.N):
                r_m = numpy.concatenate((r_m, r_matrix_n[(lin, col)]), axis=1)
                g_m = numpy.concatenate((g_m, g_matrix_n[(lin, col)]), axis=1)
                b_m = numpy.concatenate((b_m, b_matrix_n[(lin, col)]), axis=1)
            if r_matrix_nn is None:
                r_matrix_nn = r_m
                g_matrix_nn = g_m
                b_matrix_nn = b_m
            else:
                r_matrix_nn = numpy.concatenate((r_matrix_nn, r_m), axis=0)
                g_matrix_nn = numpy.concatenate((g_matrix_nn, g_m), axis=0)
                b_matrix_nn = numpy.concatenate((b_matrix_nn, b_m), axis=0)

        coeffs_r = dwt2(r_matrix_nn, self.WAVELET)
        cA1r, (cH1r, cV1r, cD1r) = coeffs_r

        SH1_r = TwoDimensionalDCT.forward(cH1r)
        SH2_r = TwoDimensionalDCT.forward(cV1r)
        SH3_r = TwoDimensionalDCT.forward(cD1r)

        coeffs_g = dwt2(g_matrix_nn, self.WAVELET)
        cA1g, (cH1g, cV1g, cD1g) = coeffs_g

        SH1_g = TwoDimensionalDCT.forward(cH1g)
        SH2_g = TwoDimensionalDCT.forward(cV1g)
        SH3_g = TwoDimensionalDCT.forward(cD1g)


        coeffs_b = dwt2(b_matrix_nn, self.WAVELET)
        cA1b, (cH1b, cV1b, cD1b) = coeffs_b

        SH1_b = TwoDimensionalDCT.forward(cH1b)
        SH2_b = TwoDimensionalDCT.forward(cV1b)
        SH3_b = TwoDimensionalDCT.forward(cD1b)

        return [SH1_r,SH2_r,SH3_r, cA1r], [SH1_g, SH2_g,SH3_g, cA1g], [SH1_b,SH2_b,SH3_b, cA1b]



    def load_watermark(self, watermark_file):
        with open(watermark_file, 'r') as fd:
            return numpy.array(json.loads(fd.read()))

    def export_image(self, image, image_file):
        name = self.get_watermark_name(image_file)
        l = image.tolist()
        with open(name, 'w+') as fd:
            json.dump(l, fd)

    def get_watermark_name(self, filename):
        filename = os.path.basename(filename)
        without_extension = os.path.splitext(filename)[0]
        return os.path.join(PROJECT_CODE_DIRECTORY, 'wm-img', without_extension + '_wm.json')
