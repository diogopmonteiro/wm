from core.algs.abstract import Algorithm
from PIL import Image
from core.algs.utils import TwoDimensionalDCT
import numpy


class Cox(Algorithm):

    mu = 127
    sigma = 255
    alpha = 1  # switch as needed later
    name = ''

    def compare(self, original_watermark, extracted_watermark):
        pass

    def embed_specific(self, image, image_file, watermark=None):
        '''
            This doesn't do none, yet.
        '''

        # Compute DCT
        fDct = TwoDimensionalDCT.forward(image)
        # Sort DCT
        sizeDCT = int(fDct.__len__()*0.1)
        sortedDCTindexes = fDct.ravel().argsort()[-sizeDCT:]
        sortedDCTindexes = (numpy.unravel_index(indx, fDct.shape) for indx in sortedDCTindexes)
        sortedUnraveled = [(fDct[indx], indx) for indx in sortedDCTindexes]

        # Get size of watermark
        if watermark is not None:
            watermark_image = Image.open(watermark)
            wm_aux = watermark_image.size()
            nbits = wm_aux[0]*wm_aux[1]
            for i in range(nbits):
                fDct[sortedUnraveled[i][1]] = fDct[sortedUnraveled[i][1]] *(TwoDimensionalDCT().forward(watermarkImage)[sortedDCTindexes[i]][i])
        else:
            nbits = numpy.random.normal(self.mu, self.sigma, sizeDCT)
        # Construct the Watermark
            for i in range(len(nbits)):
                fDct[sortedUnraveled[i][1]] += self.alpha * nbits[i]
        self.export_image(sortedUnraveled, image_file, watermark)
        inverse = TwoDimensionalDCT.inverse(fDct)
        return inverse

    def extract_specific(self, image, watermark):
        pass

    def export_image(self, unraveled_arr, image_file, wm=None):
        import json, os
        name = image_file[:-4] + '_wm.json'
        dict_save = {}
        print unraveled_arr
        for i in range(len(unraveled_arr)):
            dict_save[str(unraveled_arr[i][1])] = str(unraveled_arr[i][0])
        print(dict_save)
        with open(name, 'w+') as fd:
            json.dump(dict_save, fd)
