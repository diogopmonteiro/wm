from core.algs.abstract import Algorithm
from core.algs.utils import TwoDimensionalDCT
import numpy
from core.settings import PROJECT_CODE_DIRECTORY
import json
import os


class Cox(Algorithm):

    INDEX_KEY = "index"
    ORIGINAL_VALUE_KEY = "original"
    INSERTED_WATERMARK_VALUE_KEY = "wm"

    mu = 127
    sigma = 255
    alpha = 1  # switch as needed later
    watermark_size_percentage = 0.1
    name = ''

    def compare(self, original_watermark, extracted_watermark):
        pass

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
            f_dct[sorted_unraveled[i][1]] += self.alpha * nbits[i]

        self.export_image(sorted_unraveled, image_file, nbits)
        inverse = TwoDimensionalDCT.inverse(f_dct)
        return inverse

    def extract_specific(self, image, watermark):
        pass

    def load_watermark(self, watermark_file):
        with open(watermark_file, 'r') as fd:
            return json.loads(fd.read())

    def export_image(self, unraveled_arr, image_file, distribution):
        filename = os.path.basename(image_file)
        without_extension = os.path.splitext(filename)[0]
        name = os.path.join(PROJECT_CODE_DIRECTORY, 'img', without_extension + '_wm.json')
        l = []
        for i in range(len(unraveled_arr)):
            entry = dict()
            entry[self.INSERTED_WATERMARK_VALUE_KEY] = distribution[i]
            entry[self.ORIGINAL_VALUE_KEY] = unraveled_arr[i][0]
            entry[self.INDEX_KEY] = list(unraveled_arr[i][1])  # list to convert to JSON
            l.append(entry)
        with open(name, 'w+') as fd:
            json.dump(l, fd)
