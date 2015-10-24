from core.algs.abstract import *
import os
import argparse


class WmError(BaseException):
    pass


class Command(object):

    @classmethod
    def validate_arguments(cls, *args):
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group()
        group.add_argument("-e", "--embed", action="store_true", help="embed a watermark into an image")
        group.add_argument("-x", "--extract", action="store_true", help="extract a watermark from an image")

        parser.add_argument("-a", "--algorithm", required=True,
                            help="the algorithm used for the embedding or extraction.")

        parser.add_argument("image_file", help="absolute or relative path to the image file to put a watermark on.")
        parser.add_argument("watermark", help="path to the watermark file")

        args = parser.parse_args()

        if not Algorithm.is_algorithm_available(args.algorithm):
            raise WmError("The algorithm " + args.algorithm + " is not available.")

        if not os.path.isfile(args.image_file):
            raise WmError("The provided image file " + args.image_file + " is not a file or does not exist")

        action = "embed" if args.embed else "extract"
        return action, Algorithm.get_instance(args.algorithm), args.image_file, args.watermark

    @classmethod
    def execute(cls, *args):
        action, algorithm, image_file, watermark = cls.validate_arguments(*args)
        method = getattr(algorithm, action)
        method(image_file)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def execute_command():
    try:
        import sys
        Command.execute(*sys.argv[1:])
    except WmError as w:
        print bcolors.FAIL + bcolors.BOLD + "WmError: " + bcolors.ENDC + bcolors.BOLD + w.message + bcolors.ENDC



