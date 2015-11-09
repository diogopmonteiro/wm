from PIL import Image
from core.algs.abstract import Algorithm
from core.algs.cox import Cox
from core.algs.utils import Metrics
from core.settings import IMAGE_DIRECTORY
import os
import time
from PIL import ImageFilter


class BenchmarkResults(object):
    TIME_KEY = 'time'
    GAMMA_KEY = 'gamma'
    PSNR_KEY = 'psnr'
    ATTACKS_KEY = 'attacks'

    def __init__(self):
        self.results = dict()

    def add_algorithm(self, algorithm, image, time, psnr):
        """
        Adds a benchmark result to the set of benchmarks results
        :param algorithm: the used algorithm
        :param image: the image
        :param time: the time to watermark
        :param nc:
        :return:
        """
        self.results[algorithm] = dict()
        self.results[algorithm][image] = dict()
        self.results[algorithm][image][self.TIME_KEY] = time
        self.results[algorithm][image][self.PSNR_KEY] = psnr
        self.results[algorithm][image][self.ATTACKS_KEY] = dict()

    def add_attack_result(self, algorithm, image, attack, time, gamma):
        """
        :param algorithm: the used algorithm
        :param image: image
        :param attack: the attack applied to the watermarked image
        :param time: time to extract
        :param gamma: correlation
        :return:
        """
        self.results[algorithm][image][self.ATTACKS_KEY][attack] = dict()
        self.results[algorithm][image][self.ATTACKS_KEY][attack][self.TIME_KEY] = time
        self.results[algorithm][image][self.ATTACKS_KEY][attack][self.GAMMA_KEY] = gamma

    def dump(self):
        for alg in self.results:
            for image in self.results[alg]:
                print "Time to embed wm: %ss" % str(self.results[alg][image][self.TIME_KEY])
                print "PSNR of WM: %s" % str(self.results[alg][image][self.PSNR_KEY])
                for attack in self.results[alg][image][self.ATTACKS_KEY]:
                    print "Attack %s has gamma %s" % (attack,
                                                      str(self.results[alg][image]
                                                          [self.ATTACKS_KEY][attack][self.GAMMA_KEY]))

class Benchmarks(object):

    DWT_DEFAULT_WM = ""

    @staticmethod
    def no_attack(image):
        return image

    @staticmethod
    def blur(image):
        return image.filter(ImageFilter.GaussianBlur(radius=5))

    @staticmethod
    def jpeg_compression(image, quality=50):
        import StringIO
        buffer = StringIO.StringIO()
        image.save(buffer, format="JPEG", quality=quality)
        return Image.open(buffer)

    def __init__(self):
        self.results = BenchmarkResults()

        self.attack_modifiers = [
            Benchmarks.blur,
            Benchmarks.jpeg_compression,
            Benchmarks.no_attack
        ]

        self.algorithms = []
        self.images = []
        self.load_algorithms()
        self.load_img_files()

    def load_algorithms(self):
        for algorithm in Algorithm.available_algorithms:
            self.algorithms.append(Algorithm.get_instance(algorithm))

    def load_img_files(self):
        self.images = [os.path.join(IMAGE_DIRECTORY,f)
                       for f in os.listdir(IMAGE_DIRECTORY)
                       if os.path.isfile(os.path.join(IMAGE_DIRECTORY, f))]

    def run(self):
        algorithm = Cox()
        image = self.images[0]
        start = time.clock()
        iw, psnr = algorithm.embed(image)
        t = (time.clock() - start)
        self.results.add_algorithm(algorithm.get_algorithm_name(), image, t, psnr)
        for attack in self.attack_modifiers:
            attacked = attack(Image.fromarray(iw))
            start = time.clock()
            wmark, gamma = algorithm.extract_specific(attacked, algorithm.get_watermark_name(image))
            t = (time.clock() - start)
            self.results.add_attack_result(algorithm.get_algorithm_name(), image, attack.__name__, t, gamma)

        self.results.dump()
        # for algorithm in self.algorithms:
        #     if algorithm.get_algorithm_name() == "DWT":
        #         continue
        #     for image in self.images:
        #         start = time.clock()
        #         iw, psnr = algorithm.embed(image)
        #         seconds = (time.clock() - start) * 1000
        #         self.results.add_algorithm(algorithm.get_algorithm_name(), image, seconds, psnr)
        #         for attack in self.attack_modifiers:
        #             attacked = attack(iw)
        #             wmark, gamma = algorithm.extract_specific(attacked)
        #             self.results.add_attack_result()