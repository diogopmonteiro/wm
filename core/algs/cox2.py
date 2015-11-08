from cox import Cox
from core.algs.utils import TwoDimensionalDCT
import numpy


class Cox(Cox):

    def embed_specific(self, image, image_file, watermark=None):
        # Compute DCT
        f_dct = TwoDimensionalDCT.forward(image)

        # Sort DCT
        size_dct = int(f_dct.__len__()*self.watermark_size_percentage)
        sorted_dct_indexes = f_dct.ravel().argsort()[-size_dct:]
        sorted_dct_indexes = (numpy.unravel_index(indx, f_dct.shape) for indx in sorted_dct_indexes)
        sorted_unraveled = [(f_dct[indx], indx) for indx in sorted_dct_indexes]

        nbits = numpy.random.normal(self.mu, self.sigma, size_dct)

        # Construct the Watermark
        for i in range(len(nbits)):
            f_dct[sorted_unraveled[i][1]] += (self.alpha * nbits[i])*f_dct[sorted_unraveled[i][1]]

        self.export_image(sorted_unraveled, image_file, nbits)
        inverse = TwoDimensionalDCT.inverse(f_dct)
        return inverse

    def extract_specific(self, image, watermark):
        f_dct = TwoDimensionalDCT.forward(image)
        w = self.load_watermark(watermark)

        xi = []
        xo = []

        for entry in w:
            xo.append( entry[self.INSERTED_WATERMARK_VALUE_KEY] )
            xi.append( ( (f_dct[tuple(entry[self.INDEX_KEY])]/entry[self.ORIGINAL_VALUE_KEY])- 1) /\
                       (self.alpha))

        print (self.calculate_gamma(xi, xo))