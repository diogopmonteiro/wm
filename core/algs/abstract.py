from pydoc import locate
from PIL import Image
import numpy


class Algorithm(object):

    WATERMARKED_IMAGES_DIRECTORY = "wm-img"

    # Dictionary that relates algorithm names (provided via command line)
    # and the class type where the algorithm is implemented.
    available_algorithms = {
        'cox': 'core.algs.cox.Cox',
        'etc': None
    }

    @classmethod
    def is_algorithm_available(cls, s):
        return s in cls.available_algorithms.keys()

    @classmethod
    def get_instance(cls, s):
        """
            Retrieve an algorithm instance from a string.
        """
        return locate(cls.available_algorithms[s])()

    def _open_image(self, file):
        img_obj = Image.open(file)
        return numpy.array(img_obj, dtype=numpy.float)

    def get_image_output_file(self, image_file):
        import os
        _, filename = os.path.split(image_file)

        if not os.path.exists(self.WATERMARKED_IMAGES_DIRECTORY):
            os.mkdir(self.WATERMARKED_IMAGES_DIRECTORY)

        return os.path.join(self.WATERMARKED_IMAGES_DIRECTORY, filename)

    def embed(self, image_file, watermark=None):
        array = self._open_image(image_file)

        changed_image = self.embed_specific(array)

        # color values range from 0 and 255 and must be integer
        changed_image = changed_image.clip(0, 255)
        changed_image = changed_image.astype('uint8')

        img = Image.fromarray(changed_image)
        img.save(self.get_image_output_file(image_file))

    def extract(self):
        pass

    def embed_specific(self, image, watermark=None):
        raise NotImplementedError("You must subclass this and implement the embed mechanism per algorithm")

    def extract_specific(self, image, watermark):
        raise NotImplementedError("You must subclass this and implement the extract mechanism per algorithm")

    #def compare(self, original_watermark, extracted_watermark):
    #    raise NotImplementedError("You must subclass this")





