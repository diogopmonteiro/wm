from core.algs.abstract import Algorithm
from PIL import Image
from core.algs.utils import TwoDimensionalDCT
import numpy


class Cox(Algorithm):

    mu = 127
    sigma = 255

    def compare(self, original_watermark, extracted_watermark):
        pass

    def embed_specific(self, image, watermark=None):
        '''
            This doesn't do none, yet.
        '''

        # Compute DCT
        fDct = TwoDimensionalDCT.forward(image)
        # Sort DCT
        sizeDCT = int(fDct.__len__()*0.1)
        sortedDCTindexes = fDct.ravel().argsort()[-sizeDCT:]
        sortedDCTindexes = (numpy.unravel_index(indx, fDct.shape) for indx in sortedDCTindexes)
        # sortedUnraveled = []
        # for i in sortedDCTindexes:
        #     sortedUnraveled.append((sortedDCTindexes[i], i))
        sortedUnraveled = [(fDct[indx], indx) for indx in sortedDCTindexes]
        # Get size of watermark
        if(watermark!=None):
            watermarkImage = Image.open(watermark)
            wmAux = watermarkImage.size()
            nbits = wmAux[0]*wmAux[1]
            for i in range(nbits):
                # I am so lost...
                fDct[sortedUnraveled[i][1]] = fDct[sortedUnraveled[i][1]] *(TwoDimensionalDCT().forward(watermarkImage)[sortedDCTindexes[i]][i])
        else:
            imgSize = int(fDct.__len__())
            nbits = numpy.random.normal(self.mu, self.sigma, sizeDCT)
        # Construct the Watermark
            for i in range(len(nbits)):
                fDct[sortedUnraveled[i][1]] = fDct[sortedUnraveled[i][1]] * nbits[i]
        inverse = TwoDimensionalDCT.inverse(fDct)
        return inverse

    def extract_specific(self, image, watermark):
        pass