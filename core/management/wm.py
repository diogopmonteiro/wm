from core.algs.abstract import *
import os


class WmError(BaseException):
    pass


class Command(object):

    actions = {
        '--embed': 'embed',
        '--extract': 'extract'
    }

    @classmethod
    def validate_arguments(cls, *args):
        if len(args) < 3:
            raise WmError("Incorrect args length.")

        action = args[0]
        if action not in cls.actions.keys():
            raise WmError("The action " + action + " is not available.")

        algorithm = args[1]
        if not Algorithm.is_algorithm_available(algorithm):
            raise WmError("The algorithm " + algorithm + " is not available.")

        image_file = args[2]
        if not os.path.isfile(image_file):
            raise WmError("The provided image file " + image_file + " is not a file or does not exist")

        return cls.actions[action], Algorithm.get_instance(algorithm), image_file

    @classmethod
    def execute(cls, *args):
        action, algorithm, image_file = cls.validate_arguments(*args)
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



