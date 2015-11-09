from core.algs.abstract import Algorithm
from core.settings import IMAGE_DIRECTORY
import os

class Benchmarks(object):

    @staticmethod
    def blur(image):
        return None

    @staticmethod
    def jpeg_compression(image, quality=50):
        return None

    def __init__(self):
        self.attack_modifiers = [
            Benchmarks.blur,
            Benchmarks.jpeg_compression
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
        for algorithm in self.algorithms:
            for attack in self.attack_modifiers:
                for image in self.images:
                    pass
