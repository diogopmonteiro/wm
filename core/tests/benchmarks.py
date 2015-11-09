import random
import numpy
from PIL import Image
from core.algs.abstract import Algorithm
from core.settings import IMAGE_DIRECTORY
import os
import time
from PIL import ImageFilter
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




class Benchmarks(object):

    @staticmethod
    def no_attack(image):
        return image

    DWT_DEFAULT_WM = ""

    @staticmethod
    def unsharp_mask(image):
        return image.filter(ImageFilter.UnsharpMask)

    @staticmethod
    def mode_filter(image):
        return image.filter(ImageFilter.ModeFilter)

    @staticmethod
    def rotate(image, rotation_degree=2):
        return image.rotate(rotation_degree)

    @staticmethod
    def median_filter(image):
        return image.filter(ImageFilter.MedianFilter)

    @staticmethod
    def noise(image, noise_level=10):
        def noise_map(i):
            noise = random.randint(0, noise_level) - noise_level / 2
            return max(0, min(i + noise, 255))
        m = numpy.vectorize(noise_map)
        result = m(numpy.array(image, dtype=numpy.float))
        result = result.astype('uint8')
        return Image.fromarray(result)

    @staticmethod
    def blur(image):
        return image.filter(ImageFilter.GaussianBlur(radius=5))

    @staticmethod
    def jpeg_compression(image, quality=30):
        import StringIO
        buffer = StringIO.StringIO()
        image.save(buffer, format="JPEG", quality=quality)
        return Image.open(buffer)

    def __init__(self, algorithm, image):
        self.algorithm = Algorithm.get_instance(algorithm)
        self.results = BenchmarkResults(self.algorithm.get_algorithm_name())

        self.attack_modifiers = [
            Benchmarks.no_attack,
            Benchmarks.blur,
            Benchmarks.jpeg_compression,
            Benchmarks.noise,
            Benchmarks.rotate,
            Benchmarks.median_filter,
            Benchmarks.mode_filter,
            Benchmarks.unsharp_mask
        ]

        self.images = []
        if image is not None:
            self.images.append(image)
        else:
            self.load_img_files()

    def load_img_files(self):
        self.images = [os.path.join(IMAGE_DIRECTORY,f)
                       for f in os.listdir(IMAGE_DIRECTORY)
                       if os.path.isfile(os.path.join(IMAGE_DIRECTORY, f))]

    def run(self):
        for image in self.images:
            start = time.clock()
            iw, psnr = self.algorithm.embed(image)
            t = (time.clock() - start)
            self.results.init_benchmarks(image, t, psnr)
            for attack in self.attack_modifiers:
                attacked = attack(Image.fromarray(iw))
                start = time.clock()
                wmark, gamma = self.algorithm.extract_specific(attacked, self.algorithm.get_watermark_name(image))
                t = (time.clock() - start)
                self.results.add_attack_result(self.algorithm.get_algorithm_name(), image, attack.__name__, t, gamma)

        self.results.dump()
