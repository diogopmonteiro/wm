from core.algs.abstract import *
from core.management import bcolors
from core.tests.benchmarks import Benchmarks
import os
import argparse


class WmError(BaseException):
    pass


class Command(object):

    @classmethod
    def add_extract_embed_args(cls, parser):
        parser.add_argument("-a", "--algorithm", required=True,
                            help="the algorithm used for the embedding or extraction.")

        parser.add_argument("image_file", help="absolute or relative path to the image file to put a watermark on.")
        parser.add_argument("watermark", nargs="?", default=None, help="path to the watermark file")

    @classmethod
    def validate_arguments(cls):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        bench = subparsers.add_parser('benchmarks', help="run benchmarks")
        bench.add_argument("-a", "--algorithm", required=True,
                            help="the algorithm used for the embedding or extraction.")
        bench.add_argument("image_file", nargs="?", default=None,
                           help="absolute or relative path to the image file to put a watermark on.")

        subparsers.add_parser('tests', help="run tests")
        embed = subparsers.add_parser('embed', help="embed a watermark into an image")
        cls.add_extract_embed_args(embed)
        extract = subparsers.add_parser('extract', help="extract a watermark from an image")
        cls.add_extract_embed_args(extract)

        return parser.parse_args()

    @classmethod
    def execute(cls):
        args = cls.validate_arguments()
        command = args.command

        if command == 'benchmarks':
            if not Algorithm.is_algorithm_available(args.algorithm):
                raise WmError("The algorithm \"" + args.algorithm + "\" is not available.")
            Benchmarks(args.algorithm, args.image_file).run()
        elif command == 'tests':
            pass
        elif command == 'embed' or command == 'extract':
            if not Algorithm.is_algorithm_available(args.algorithm):
                raise WmError("The algorithm \"" + args.algorithm + "\" is not available.")

            if not os.path.isfile(args.image_file):
                raise WmError("The provided image file \"" + args.image_file + "\" is not a file or does not exist.")

            if args.watermark and not os.path.exists(args.watermark):
                raise WmError("The provided watermark file \"" + args.watermark + "\" is not a file or does not exist.")

            action = "embed" if command == 'embed' else "extract"
            algorithm, image_file, watermark = Algorithm.get_instance(args.algorithm), args.image_file, args.watermark
            method = getattr(algorithm, action)
            method(image_file, watermark)


def execute_command():
    try:
        Command.execute()
    except WmError as w:
        print bcolors.FAIL + bcolors.BOLD + "WmError: " + bcolors.ENDC + bcolors.BOLD + w.message + bcolors.ENDC



