from core.algs.abstract import Algorithm
from core.settings import PROJECT_CODE_DIRECTORY
import json
import os
import math


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


    def calculate_gamma(self, w, o):
        top = 0
        bw = 0
        bo = 0
        for i in range(len(w)):
            top += w[i]*o[i]
            bw += w[i]*w[i]
            bo += o[i] * o[i]

        return (top)/(math.sqrt(bw)*math.sqrt(bo))

    def load_watermark(self, watermark_file):
        with open(watermark_file, 'r') as fd:
            return json.loads(fd.read())

    def export_image(self, unraveled_arr, image_file, distribution):
        filename = os.path.basename(image_file)
        without_extension = os.path.splitext(filename)[0]
        name = os.path.join(PROJECT_CODE_DIRECTORY, 'wm-img', without_extension + '_wm.json')
        l = []
        for i in range(len(unraveled_arr)):
            entry = dict()
            entry[self.INSERTED_WATERMARK_VALUE_KEY] = distribution[i]
            entry[self.ORIGINAL_VALUE_KEY] = unraveled_arr[i][0]
            entry[self.INDEX_KEY] = list(unraveled_arr[i][1])  # list to convert to JSON
            l.append(entry)
        with open(name, 'w+') as fd:
            json.dump(l, fd)
