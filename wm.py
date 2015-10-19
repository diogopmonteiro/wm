import sys
from algs.abstract import get_algorithm_instance
from PIL import Image
import numpy

'''
    python wm.py --extract cox image.file watermark-file
    python wm.py --embed cox image.file
'''

def open_image(file):
    img_obj = Image.open(file)
    return numpy.array(img_obj, dtype=numpy.float)


def template_embed(algorithm, image_file):
    output = 'wm-' + image_file

    array = open_image(image_file)

    changed_image = algorithm.embed(array)

    #color values range from 0 and 255 and must be integer
    changed_image = changed_image.clip(0, 255)
    changed_image = changed_image.astype('uint8')

    img = Image.fromarray(changed_image)
    img.save(output)


def template_extract():
    pass


actions = {
    '--embed': template_embed,
    '--extract': template_extract
}



def boot():
    if len(sys.argv) <= 2:
        return "You must select an algorithm and an image"

    argv = sys.argv[1:]
    action = argv[0]
    algorithm = argv[1]
    image_file = argv[2]
    actions[action](get_algorithm_instance(algorithm), image_file)


boot()
