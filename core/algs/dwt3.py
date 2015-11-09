from core.algs.abstract import Algorithm


class DWT(Algorithm):

    WAVELET = 'db1'

    def embed_specific(self, image, image_file, watermark=None):
        pass

    def extract_specific(self, image, watermark):
        pass
