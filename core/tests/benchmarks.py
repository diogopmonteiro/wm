import random
from core.algs.utils import Metrics
import numpy
from PIL import Image
from core.algs.abstract import Algorithm
from core.settings import IMAGE_DIRECTORY, WM_DIRECTORY
import os
import time
from PIL import ImageFilter, ImageEnhance
from core.management import bcolors


class BenchmarkResults(object):
    TIME_KEY = 'time'
    GAMMA_KEY = 'gamma'
    PSNR_KEY = 'psnr'
    ATTACKS_KEY = 'attacks'

    HORIZONTAL_LENGTH_CHARS = 60

    def __init__(self, algorithm_name):
        self.results = dict()
        self.algorithm_name = algorithm_name

    def init_benchmarks(self, image, time, psnr):
        """
        Adds a benchmark result to the set of benchmarks results
        :param algorithm: the used algorithm
        :param image: the image
        :param time: the time to watermark
        :param nc:
        :return:
        """
        self.results[image] = dict()
        self.results[image][self.TIME_KEY] = time
        self.results[image][self.PSNR_KEY] = psnr
        self.results[image][self.ATTACKS_KEY] = dict()

    def add_attack_result(self, algorithm, image, attack, time, gamma):
        """
        :param algorithm: the used algorithm
        :param image: image
        :param attack: the attack applied to the watermarked image
        :param time: time to extract
        :param gamma: correlation
        :return:
        """
        self.results[image][self.ATTACKS_KEY][attack] = dict()
        self.results[image][self.ATTACKS_KEY][attack][self.TIME_KEY] = time
        self.results[image][self.ATTACKS_KEY][attack][self.GAMMA_KEY] = gamma

    def get_char_to_center(self, char, length, string):
        return char * ((length - len(string))/2)

    def dump(self):
        """
            Prints benchmarks results to stdout.
        """
        s = "+%s%s%s+" % \
              (self.get_char_to_center("-", self.HORIZONTAL_LENGTH_CHARS, self.algorithm_name),
               self.algorithm_name,
               self.get_char_to_center("-", self.HORIZONTAL_LENGTH_CHARS, self.algorithm_name))

        print bcolors.OKBLUE + bcolors.BOLD + s + bcolors.ENDC

        for image in self.results:
            table = [["Attack", "Gamma", "Time"]]
            print "+---> Image %s" % os.path.split(image)[1]
            print "\tTime to embed watermark: %ss" % str(self.results[image][self.TIME_KEY])
            print "\tPSNR: %s" % str(self.results[image][self.PSNR_KEY])
            for attack in self.results[image][self.ATTACKS_KEY]:
                table.append([attack,
                              self.results[image][self.ATTACKS_KEY][attack][self.GAMMA_KEY],
                              self.results[image][self.ATTACKS_KEY][attack][self.TIME_KEY]
                              ])
            import tabulate
            print tabulate.tabulate(table, tablefmt="grid")


def attackmethod(func):
    """
    Add this decorator to a function that implements an attack to an image
    """
    func.is_attack = True
    return func


class Benchmarks(object):

    @staticmethod
    @attackmethod
    def no_attack(image):
        return image

    DWT_DEFAULT_WM = ""

    @staticmethod
    @attackmethod
    def add_contrast(image):
        return ImageEnhance.Contrast(image).enhance(1.2)

    @staticmethod
    @attackmethod
    def unsharp_mask(image):
        return image.filter(ImageFilter.UnsharpMask)

    @staticmethod
    @attackmethod
    def mode_filter(image):
        return image.filter(ImageFilter.ModeFilter)

    @staticmethod
    @attackmethod
    def rotate(image, rotation_degree=3):
        return image.rotate(rotation_degree)

    @staticmethod
    @attackmethod
    def median_filter(image):
        return image.filter(ImageFilter.MedianFilter)

    @staticmethod
    @attackmethod
    def noise(image, noise_level=20):
        def noise_map(i):
            noise = random.randint(0, noise_level) - noise_level / 2
            return max(0, min(i + noise, 255))
        m = numpy.vectorize(noise_map)
        result = m(numpy.array(image, dtype=numpy.float))
        result = result.astype('uint8')
        return Image.fromarray(result)

    @staticmethod
    @attackmethod
    def blur(image):
        return image.filter(ImageFilter.GaussianBlur(radius=5))

    @staticmethod
    @attackmethod
    def jpeg_compression(image, quality=10):
        import StringIO
        buffer = StringIO.StringIO()
        image.save(buffer, format="JPEG", quality=quality)
        return Image.open(buffer)

    @staticmethod
    def crop_general(image, size):
        '''
        Size can be predetermined as a variable % of the picture size
        '''
        width, height = image.size
        left = numpy.ceil((width - width*size)/2)
        top = numpy.ceil((height - height*size)/2)
        right = numpy.floor((width + width*size)/2)
        bottom = numpy.floor((height + height*size)/2)

        image.paste((255, 255, 255), (int(left), int(top), int(right), int(bottom)))
        return image

    @staticmethod
    @attackmethod
    def crop_picture_small(image):
        return Benchmarks.crop_general(image, 0.05)

    @staticmethod
    @attackmethod
    def crop_picture_middle(image):
        return Benchmarks.crop_general(image, 0.15)

    @staticmethod
    @attackmethod
    def crop_picture_big(image):
        return Benchmarks.crop_general(image, 0.3)

    @staticmethod
    @attackmethod
    def size_up_down(image):
        w_ori, h_ori = image.size
        w_new, h_new = int(w_ori*1.5), int(h_ori*1.5)
        im = image.resize((w_new, h_new))
        return im.resize((w_ori, h_ori))

    @staticmethod
    @attackmethod
    def size_down_up(image):
        w_ori, h_ori = image.size
        w_new, h_new = int(w_ori*0.75), int(h_ori*0.75)
        im = image.resize((w_new, h_new))
        return im.resize((w_ori, h_ori))

    def load_attack_modifiers(self):
        import types
        results = list()
        for attr in Benchmarks.__dict__:
            obj_attr = getattr(Benchmarks, attr)

            if isinstance(obj_attr, types.FunctionType) and hasattr(obj_attr, 'is_attack'):
                results.append(obj_attr)
        return results

    def __init__(self, algorithm, image, watermark_file):
        self.algorithm = Algorithm.get_instance(algorithm)
        self.results = BenchmarkResults(self.algorithm.get_algorithm_name())
        self.watermark_file = watermark_file
        self.attack_modifiers = self.load_attack_modifiers()

        self.images = []
        if image is not None:
            self.images.append(image)
        else:
            self.load_img_files()

    def load_img_files(self):
        """
            Load all image paths in the images directory to memory.
        """
        self.images = [os.path.join(IMAGE_DIRECTORY,f)
                       for f in os.listdir(IMAGE_DIRECTORY)
                       if os.path.isfile(os.path.join(IMAGE_DIRECTORY, f))]

    def run(self):
        # Create watermark directory and algorithm directory in case they don't still exist
        if not os.path.exists(WM_DIRECTORY):
            os.mkdir(WM_DIRECTORY)
        path = os.path.join(WM_DIRECTORY, self.algorithm.get_algorithm_name())
        if not os.path.exists(path):
            os.mkdir(path)

        for image in self.images:
            path = os.path.join(path, os.path.split(image)[1] + "-benchmarks")
            if not os.path.exists(path):
                os.mkdir(path)

            start = time.clock()
            iw, psnr = self.algorithm.embed(image, self.watermark_file)
            t = (time.clock() - start)

            self.results.init_benchmarks(image, t, psnr)

            # Apply attacks
            for attack in self.attack_modifiers:
                attacked = attack(Image.fromarray(iw))
                attacked.save(os.path.join(path, attack.__name__ + "-" + os.path.split(image)[1]))

                start = time.clock()
                if self.algorithm.get_algorithm_name() == "Cox":
                    wmark, gamma = self.algorithm.extract_specific(attacked, self.algorithm.get_watermark_name(image))
                elif self.algorithm.get_algorithm_name() == "DWT":
                    wmark, gamma = self.algorithm.extract_specific(numpy.array(Image.open(image)),
                                                          os.path.join(path, attack.__name__ + "-" + os.path.split(image)[1]))
                    wmark = wmark.clip(0,255)
                    wmark = wmark.astype('uint8')
                    gamma = Metrics.gamma(wmark.ravel(), numpy.array(Image.open(self.watermark_file)).ravel())
                    wmark = Image.fromarray(wmark)
                    wmark.save(os.path.join(path, attack.__name__ + "-extracted-wm-" + os.path.split(image)[1]))

                t = (time.clock() - start)
                self.results.add_attack_result(self.algorithm.get_algorithm_name(), image, attack.__name__, t, gamma)

        self.results.dump()
