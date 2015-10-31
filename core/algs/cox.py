import numpy
from core.algs.abstract import Algorithm
from PIL import Image
from core.algs.utils import TwoDimensionalDCT
import numpy


class Cox(Algorithm):

    def compare(self, original_watermark, extracted_watermark):
        pass

    def embed_specific(self, image, watermark=None):
        '''
            This doesn't do none, yet.
        '''

        # Compute DCT
        fDct = TwoDimensionalDCT.forward(image)
        # Sort DCT
        sortedDCTindexes = numpy.argsort(fDct)[::-1]
        # Get size of watermark
        if(watermark!=None):
            watermarkImage= Image.open(watermark)
            wmAux = watermarkImage.size()
            nbits = wmAux[0]*wmAux[1]
            for i in range(nbits):
                # I am so lost...
                fDct[i][sortedDCTindexes[i]] = fDct[i][sortedDCTindexes[i]] * \
                                               (TwoDimensionalDCT().forward(watermarkImage)[sortedDCTindexes[i]][i])
        else:
            imgSize = int(fDct.__len__())
            nbits = numpy.random.normal(127, 255, imgSize)
        # Construct the Watermark
            for i in range(len(nbits)):
                fDct[i][sortedDCTindexes[i]] = fDct[i][sortedDCTindexes[i]] * nbits[i]
        inverse = TwoDimensionalDCT.inverse(fDct)
        return image

    def extract_specific(self, image, watermark):
        pass