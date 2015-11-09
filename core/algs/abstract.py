from pydoc import locate
from PIL import Image
from core.algs.utils import Metrics
import numpy
import os


class Algorithm(object):

    WATERMARKED_IMAGES_DIRECTORY = "wm-img"

    # Dictionary that relates algorithm names (provided via command line)
    # and the class type where the algorithm is implemented.
    available_algorithms = {
        'dwt':  'core.algs.dwt1.DWT',
        'cox': 'core.algs.cox.Cox',
        'dummy': 'core.algs.abstract.Dummy'
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

    def open_image(self, file):
        img_obj = Image.open(file)
        img_obj = img_obj.convert("RGB")
        return numpy.array(img_obj, dtype=numpy.float)

    def get_image_output_file(self, image_file):
        _, filename = os.path.split(image_file)

        return os.path.join(self.WATERMARKED_IMAGES_DIRECTORY, filename)

    def embed(self, image_file, watermark=None):
        if not os.path.exists(self.WATERMARKED_IMAGES_DIRECTORY):
            os.mkdir(self.WATERMARKED_IMAGES_DIRECTORY)

        array = self.open_image(image_file)

        if watermark != None:
            watermark = self.open_image(watermark)

        changed_image = self.embed_specific(array, image_file, watermark)

        # color values range from 0 and 255 and must be integer
        changed_image = changed_image.clip(0, 255)
        changed_image = changed_image.astype('uint8')

        img = Image.fromarray(changed_image)

        img.save(self.get_image_output_file(image_file))

        psnr = Metrics.psnr(numpy.array(Image.fromarray(changed_image).convert('RGB'), dtype=numpy.float),
                            numpy.array(Image.open(image_file).convert('RGB'), dtype=numpy.float))
        # such a damn workaround

        print "PSNR %s" % str(psnr)

        return changed_image, psnr

    def extract(self, image_file, watermark):
        array = self.open_image(image_file)

        wmark, gamma = self.extract_specific(array, watermark)

        if wmark is not None:
            wmark = wmark.clip(0, 255)
            wmark = wmark.astype('uint8')

            img = Image.fromarray(wmark)

            img.save()
        print "Gamma %s" % str(gamma)
        return wmark, gamma

    def embed_specific(self, image, image_file, watermark=None):
        raise NotImplementedError("You must subclass this and implement the embed mechanism per algorithm")

    def extract_specific(self, image, watermark):
        raise NotImplementedError("You must subclass this and implement the extract mechanism per algorithm")

    def get_watermark_name(self, filename):
        raise NotImplementedError("You must subclass this and implement the extract mechanism per algorithm")

    def get_algorithm_name(self):
        return self.__class__.__name__


class Dummy(Algorithm):
    def embed_specific(self, image, image_file, watermark=None):
        return image

    def extract_specific(self, image, watermark):
        return image




