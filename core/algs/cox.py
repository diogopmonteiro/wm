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
        img = Image.open(image)
        arrayImage = numpy.array(img)
        fDct = TwoDimensionalDCT.forward(arrayImage)
        # Sort DCT
        sortedDCTindexes = numpy.argsort(fDct.ravel())[::-1]
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
            imgSize = int((img.size[0]*img.size[1])*0.1)
            nbits = numpy.random.normal(127, 255, imgSize)
        # Construct the Watermark
            for i in range(len(nbits)):
                fDct[i][sortedDCTindexes[i]] = fDct[i][sortedDCTindexes[i]] * nbits[i]

        inverse = TwoDimensionalDCT.inverse(fDct)
        inverse = inverse.clip(0, 255)
        inverse = inverse.astype('uint8')
        image = Image.fromarray(inverse)

        return image

    def extract_specific(self, image, watermark):
        pass