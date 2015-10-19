from pydoc import locate


class Algorithm(object):

    def embed(self, image, watermark=None):
        raise NotImplementedError("You must subclass this")

    def extract(self, image, watermark):
        raise NotImplementedError("You must subclass this")

    def compare(self, original_watermark, extracted_watermark):
        raise NotImplementedError("You must subclass this")



'''
    Dictionary that relates algorithm names (provided via command line)
    and the class type where the algorithm is implemented.
'''
algorithms = {
    'cox': 'algs.cox.Cox',
    'etc': None
}

def get_algorithm_instance(algorithm_str):
    '''
        Retrieve an algorithm instance from a string.
    '''
    return locate(algorithms[algorithm_str])()