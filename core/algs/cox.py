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
        f_dct = TwoDimensionalDCT.forward(image)
        # Sort DCT
        size_dct = int(f_dct.__len__()*0.1)
        sorted_dct_indexes = f_dct.ravel().argsort()[-size_dct:]
        sorted_dct_indexes = (numpy.unravel_index(indx, f_dct.shape) for indx in sorted_dct_indexes)
        sorted_unraveled = [(f_dct[indx], indx) for indx in sorted_dct_indexes]

        # Get size of watermark
        if watermark is not None:
            watermark_image = Image.open(watermark)
            wm_aux = watermark_image.size()
            nbits = wm_aux[0]*wm_aux[1]
            for i in range(nbits):
                f_dct[sorted_unraveled[i][1]] = f_dct[sorted_unraveled[i][1]] * \
                                                (TwoDimensionalDCT().forward(watermark_image)[sorted_dct_indexes[i]][i])
        else:
            nbits = numpy.random.normal(self.mu, self.sigma, size_dct)
        # Construct the Watermark
            for i in range(len(nbits)):
                f_dct[sorted_unraveled[i][1]] += self.alpha * nbits[i]
        self.export_image(sorted_unraveled, image_file, watermark)
        inverse = TwoDimensionalDCT.inverse(f_dct)
        return inverse

    def extract_specific(self, image, watermark):
        pass

    def export_image(self, unraveled_arr, image_file, wm=None):
        import json
        name = image_file[:-4] + '_wm.json'
        dict_save = {}
        print unraveled_arr
        for i in range(len(unraveled_arr)):
            dict_save[str(unraveled_arr[i][1])] = str(unraveled_arr[i][0])
        print(dict_save)
        with open(name, 'w+') as fd:
            json.dump(dict_save, fd)
