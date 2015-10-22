from core.algs.abstract import Algorithm


class Cox(Algorithm):

    def compare(self, original_watermark, extracted_watermark):
        pass

    def embed_specific(self, image, watermark=None):
        '''
            This doesn't do none, yet.
        '''
        return image

    def extract_specific(self, image, watermark):
        pass