from abstract import Algorithm
from core.algs.utils import TwoDimensionalDCT
from scipy.misc import imread
from pywt import dwt2, idwt2
import numpy
from PIL import Image


class DWT(Algorithm):

    WAVELET = 'db1'
    N = 4
    alpha = 1

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

        DcH1r = TwoDimensionalDCT.forward(cH1r)
        DcV1r = TwoDimensionalDCT.forward(cV1r)
        DcD1r = TwoDimensionalDCT.forward(cD1r)

        U1r, s1r, SH1_r = numpy.linalg.svd(DcH1r)
        U2r, s2r, SH2_r = numpy.linalg.svd(DcV1r)
        U3r, s3r, SH3_r = numpy.linalg.svd(DcD1r)

        coeffs_g = dwt2(g_matrix_nn, self.WAVELET)
        cA1g, (cH1g, cV1g, cD1g) = coeffs_g

        DcH1g = TwoDimensionalDCT.forward(cH1g)
        DcV1g = TwoDimensionalDCT.forward(cV1g)
        DcD1g = TwoDimensionalDCT.forward(cD1g)

        U1g, s1g, SH1_g = numpy.linalg.svd(DcH1g)
        U2g, s2g, SH2_g = numpy.linalg.svd(DcV1g)
        U3g, s3g, SH3_g = numpy.linalg.svd(DcD1g)

        coeffs_b = dwt2(b_matrix_nn, self.WAVELET)
        cA1b, (cH1b, cV1b, cD1b) = coeffs_b

        DcH1b = TwoDimensionalDCT.forward(cH1b)
        DcV1b = TwoDimensionalDCT.forward(cV1b)
        DcD1b = TwoDimensionalDCT.forward(cD1b)

        U1b, s1b, SH1_b = numpy.linalg.svd(DcH1b)
        U2b, s2b, SH2_b = numpy.linalg.svd(DcV1b)
        U3b, s3b, SH3_b = numpy.linalg.svd(DcD1b)

        rw = watermark[:,:,2]
        gw = watermark[:,:,1]
        bw = watermark[:,:,0]

        coeffs_rw = dwt2(rw, self.WAVELET)
        cA1rw, (cH1rw, cV1rw, cD1rw) = coeffs_rw

        DcH1rw = TwoDimensionalDCT.forward(cH1rw)
        DcV1rw = TwoDimensionalDCT.forward(cV1rw)
        DcD1rw = TwoDimensionalDCT.forward(cD1rw)

        U1rw, s1rw, SH1_rw = numpy.linalg.svd(DcH1rw)
        U2rw, s2rw, SH2_rw = numpy.linalg.svd(DcV1rw)
        U3rw, s3rw, SH3_rw = numpy.linalg.svd(DcD1rw)

        coeffs_gw = dwt2(gw, self.WAVELET)
        cA1gw, (cH1gw, cV1gw, cD1gw) = coeffs_gw

        DcH1gw = TwoDimensionalDCT.forward(cH1gw)
        DcV1gw = TwoDimensionalDCT.forward(cV1gw)
        DcD1gw = TwoDimensionalDCT.forward(cD1gw)

        U1gw, s1gw, SH1_gw = numpy.linalg.svd(DcH1gw)
        U2gw, s2gw, SH2_gw = numpy.linalg.svd(DcV1gw)
        U3gw, s3gw, SH3_gw = numpy.linalg.svd(DcD1gw)

        coeffs_bw = dwt2(bw, self.WAVELET)
        cA1bw, (cH1bw, cV1bw, cD1bw) = coeffs_bw

        DcH1bw = TwoDimensionalDCT.forward(cH1bw)
        DcV1bw = TwoDimensionalDCT.forward(cV1bw)
        DcD1bw = TwoDimensionalDCT.forward(cD1bw)

        U1bw, s1bw, SH1_bw = numpy.linalg.svd(DcH1bw)
        U2bw, s2bw, SH2_bw = numpy.linalg.svd(DcV1bw)
        U3bw, s3bw, SH3_bw = numpy.linalg.svd(DcD1bw)

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

        for i in range(len(U1r)):
            for j in range(len(U1r[0])):
                U1r[i][j] += U1rw[i][j]*self.alpha
                U1g[i][j] += U1gw[i][j]*self.alpha
                U1b[i][j] += U1bw[i][j]*self.alpha
                U2r[i][j] += U2rw[i][j]*self.alpha
                U2g[i][j] += U2gw[i][j]*self.alpha
                U2b[i][j] += U2bw[i][j]*self.alpha
                U3r[i][j] += U3rw[i][j]*self.alpha
                U3g[i][j] += U3gw[i][j]*self.alpha
                U3b[i][j] += U3bw[i][j]*self.alpha

        R1 = numpy.dot(numpy.dot(U1r, numpy.diag(s1r)), SH1_r)
        R2 = numpy.dot(numpy.dot(U2r, numpy.diag(s2r)), SH2_r)
        R3 = numpy.dot(numpy.dot(U3r, numpy.diag(s3r)), SH3_r)

        iR1 = TwoDimensionalDCT.inverse(R1)
        iR2 = TwoDimensionalDCT.inverse(R2)
        iR3 = TwoDimensionalDCT.inverse(R3)

        coeffs_r = cA1r, (iR1, iR2, iR3)

        r = idwt2(coeffs_r, self.WAVELET)

        G1 = numpy.dot(numpy.dot(U1g, numpy.diag(s1g)), SH1_g)
        G2 = numpy.dot(numpy.dot(U2g, numpy.diag(s2g)), SH2_g)
        G3 = numpy.dot(numpy.dot(U3g, numpy.diag(s3g)), SH3_g)

        iG1 = TwoDimensionalDCT.inverse(G1)
        iG2 = TwoDimensionalDCT.inverse(G2)
        iG3 = TwoDimensionalDCT.inverse(G3)

        coeffs_g = cA1g, (iG1, iG2, iG3)

        g = idwt2(coeffs_g, self.WAVELET)

        B1 = numpy.dot(numpy.dot(U1b, numpy.diag(s1b)), SH1_b)
        B2 = numpy.dot(numpy.dot(U2b, numpy.diag(s2b)), SH2_b)
        B3 = numpy.dot(numpy.dot(U3b, numpy.diag(s3b)), SH3_b)

        iB1 = TwoDimensionalDCT.inverse(B1)
        iB2 = TwoDimensionalDCT.inverse(B2)
        iB3 = TwoDimensionalDCT.inverse(B3)

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

        return numpy.array(final)

    def extract_specific(self, image, watermark):

        self.info = DWTinfo()

        wm = self.open_image(watermark)

        self.info.r_matrix, self.info.g_matrix, self.info.b_matrix = self.start_op(image)

        self.info.r_matrix_o, self.info.g_matrix_o, self.info.b_matrix_o = self.start_op(wm)

        # For the watermarked

        self.info.coeffs_r = dwt2(self.info.r_matrix, self.WAVELET)
        self.info.cA1r, (self.info.cH1r, self.info.cV1r, self.info.cD1r) = self.info.coeffs_r

        self.info.DcH1r = TwoDimensionalDCT.forward(self.info.cH1r)
        self.info.DcV1r = TwoDimensionalDCT.forward(self.info.cV1r)
        self.info.DcD1r = TwoDimensionalDCT.forward(self.info.cD1r)

        self.info.U1r, self.info.s1r, self.info.SH1_r = numpy.linalg.svd(self.info.DcH1r)
        self.info.U2r, self.info.s2r, self.info.SH2_r = numpy.linalg.svd(self.info.DcV1r)
        self.info.U3r, self.info.s3r, self.info.SH3_r = numpy.linalg.svd(self.info.DcD1r)

        self.info.coeffs_g = dwt2(self.info.g_matrix, self.WAVELET)
        self.info.cA1g, (self.info.cH1g, self.info.cV1g, self.info.cD1g) = self.info.coeffs_g

        self.info.DcH1g = TwoDimensionalDCT.forward(self.info.cH1g)
        self.info.DcV1g = TwoDimensionalDCT.forward(self.info.cV1g)
        self.info.DcD1g = TwoDimensionalDCT.forward(self.info.cD1g)

        self.info.U1g, self.info.s1g, self.info.SH1_g = numpy.linalg.svd(self.info.DcH1g)
        self.info.U2g, self.info.s2g, self.info.SH2_g = numpy.linalg.svd(self.info.DcV1g)
        self.info.U3g, self.info.s3g, self.info.SH3_g = numpy.linalg.svd(self.info.DcD1g)

        self.info.coeffs_b = dwt2(self.info.b_matrix, self.WAVELET)
        self.info.cA1b, (self.info.cH1b, self.info.cV1b, self.info.cD1b) = self.info.coeffs_b

        self.info.DcH1b = TwoDimensionalDCT.forward(self.info.cH1b)
        self.info.DcV1b = TwoDimensionalDCT.forward(self.info.cV1b)
        self.info.DcD1b = TwoDimensionalDCT.forward(self.info.cD1b)

        self.info.U1b, self.info.s1b, self.info.SH1_b = numpy.linalg.svd(self.info.DcH1b)
        self.info.U2b, self.info.s2b, self.info.SH2_b = numpy.linalg.svd(self.info.DcV1b)
        self.info.U3b, self.info.s3b, self.info.SH3_b = numpy.linalg.svd(self.info.DcD1b)


        # For the original

        self.info.coeffs_r_o = dwt2(self.info.r_matrix_o, self.WAVELET)
        self.info.cA1r_o, (self.info.cH1r_o, self.info.cV1r_o, self.info.cD1r_o) = self.info.coeffs_r_o

        self.info.DcH1r_o = TwoDimensionalDCT.forward(self.info.cH1r_o)
        self.info.DcV1r_o = TwoDimensionalDCT.forward(self.info.cV1r_o)
        self.info.DcD1r_o = TwoDimensionalDCT.forward(self.info.cD1r_o)

        self.info.U1r_o, self.info.s1r_o, self.info.SH1_r_o = numpy.linalg.svd(self.info.DcH1r_o)
        self.info.U2r_o, self.info.s2r_o, self.info.SH2_r_o = numpy.linalg.svd(self.info.DcV1r_o)
        self.info.U3r_o, self.info.s3r_o, self.info.SH3_r_o = numpy.linalg.svd(self.info.DcD1r_o)

        self.info.coeffs_g_o = dwt2(self.info.g_matrix_o, self.WAVELET)
        self.info.cA1g_o, (self.info.cH1g_o, self.info.cV1g_o, self.info.cD1g_o) = self.info.coeffs_g_o

        self.info.DcH1g_o = TwoDimensionalDCT.forward(self.info.cH1g_o)
        self.info.DcV1g_o = TwoDimensionalDCT.forward(self.info.cV1g_o)
        self.info.DcD1g_o = TwoDimensionalDCT.forward(self.info.cD1g_o)

        self.info.U1g_o, self.info.s1g_o, self.info.SH1_g_o = numpy.linalg.svd(self.info.DcH1g_o)
        self.info.U2g_o, self.info.s2g_o, self.info.SH2_g_o = numpy.linalg.svd(self.info.DcV1g_o)
        self.info.U3g_o, self.info.s3g_o, self.info.SH3_g_o = numpy.linalg.svd(self.info.DcD1g_o)

        self.info.coeffs_b_o = dwt2(self.info.b_matrix_o, self.WAVELET)
        self.info.cA1b_o, (self.info.cH1b_o, self.info.cV1b_o, self.info.cD1b_o) = self.info.coeffs_b_o

        self.info.DcH1b_o = TwoDimensionalDCT.forward(self.info.cH1b_o)
        self.info.DcV1b_o = TwoDimensionalDCT.forward(self.info.cV1b_o)
        self.info.DcD1b_o = TwoDimensionalDCT.forward(self.info.cD1b_o)

        self.info.U1b_o, self.info.s1b_o, self.info.SH1_b_o = numpy.linalg.svd(self.info.DcH1b_o)
        self.info.U2b_o, self.info.s2b_o, self.info.SH2_b_o = numpy.linalg.svd(self.info.DcV1b_o)
        self.info.U3b_o, self.info.s3b_o, self.info.SH3_b_o = numpy.linalg.svd(self.info.DcD1b_o)

        # Aleluia

        for i in range(len(self.info.SH1_r)):
            for j in range(len(self.info.SH1_r[0])):
                self.info.SH1_r[i][j] = (self.info.SH1_r[i][j] - self.info.SH1_r_o[i][j]) / self.alpha
                self.info.SH1_g[i][j] = (self.info.SH1_g[i][j] - self.info.SH1_g_o[i][j]) / self.alpha
                self.info.SH1_b[i][j] = (self.info.SH1_b[i][j] - self.info.SH1_b_o[i][j]) / self.alpha
                self.info.SH2_r[i][j] = (self.info.SH2_r[i][j] - self.info.SH2_r_o[i][j]) / self.alpha
                self.info.SH2_g[i][j] = (self.info.SH2_g[i][j] - self.info.SH2_g_o[i][j]) / self.alpha
                self.info.SH2_b[i][j] = (self.info.SH2_b[i][j] - self.info.SH2_b_o[i][j]) / self.alpha
                self.info.SH3_r[i][j] = (self.info.SH3_r[i][j] - self.info.SH3_r_o[i][j]) / self.alpha
                self.info.SH3_g[i][j] = (self.info.SH3_g[i][j] - self.info.SH3_g_o[i][j]) / self.alpha
                self.info.SH3_b[i][j] = (self.info.SH3_b[i][j] - self.info.SH3_b_o[i][j]) / self.alpha

        self.info.ir1 = self.invert_band(self.info.U1r, self.info.s1r, self.info.SH1_r)
        self.info.ir2 = self.invert_band(self.info.U2r, self.info.s2r, self.info.SH2_r)
        self.info.ir3 = self.invert_band(self.info.U3r, self.info.s3r, self.info.SH3_r)

        self.info.coeffs_r = self.info.cA1r, (self.info.ir1, self.info.ir2, self.info.ir3)

        self.info.r = self.info.idwt2(self.info.coeffs_r, self.WAVELET)

        self.info.ig1 = self.invert_band(self.info.U1g, self.info.s1g, self.info.SH1_g)
        self.info.ig2 = self.invert_band(self.info.U2g, self.info.s2g, self.info.SH2_g)
        self.info.ig3 = self.invert_band(self.info.U3g, self.info.s3g, self.info.SH3_g)

        self.info.coeffs_g = self.info.cA1g, (self.info.ig1, self.info.ig2, self.info.ig3)

        self.info.g = idwt2(self.info.coeffs_g, self.WAVELET)

        self.info.ib1 = self.invert_band(self.info.U1b, self.info.s1b, self.info.SH1_b)
        self.info.ib2 = self.invert_band(self.info.U2b, self.info.s2b, self.info.SH2_b)
        self.info.ib3 = self.invert_band(self.info.U3b, self.info.s3b, self.info.SH3_b)

        self.info.coeffs_b = self.info.cA1b, (self.info.ib1, self.info.ib2, self.info.ib3)

        self.info.b = idwt2(self.info.coeffs_b, self.WAVELET)

        final = self.final_array(self.info.r, self.info.g, self.info.b)

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
        r = image[:, :, 2]
        g = image[:, :, 1]
        b = image[:, :, 0]
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
            for col in range(1,self.N):
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
        return r_matrix_nn, g_matrix_nn, b_matrix_nn

    def final_array(self, arr_r, arr_g, arr_b):
        result_array = []
        for lin in range(len(arr_r)):
            line = []
            for col in range(len(arr_r[0])):
                line.append([arr_b[lin][col], arr_g[lin][col], arr_r[lin][col]])
            result_array.append(line)
        return result_array

    def invert_band(self, u_att, s_att, sh_att):
        not_inverse = numpy.dot(numpy.dot(u_att, numpy.diag(s_att)), sh_att)
        return TwoDimensionalDCT.inverse(not_inverse)


class DWTinfo:

    def __init__(self):
        pass
