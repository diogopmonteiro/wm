from scipy import fftpack, stats
from PIL import Image
import numpy
import random


def twoDimensionalDCT(img):
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')


def inverseTwoDimensionalDCT(img):
    return fftpack.idct(fftpack.idct(img.T, norm='ortho').T, norm='ortho')


def open_image(img):
    img_obj = Image.open(img)
    return img_obj, numpy.array(img_obj, dtype=numpy.float)


def n_max(arr, n):
    indices = arr.ravel().argsort()[-n:]
    indices = (numpy.unravel_index(i, arr.shape) for i in indices)
    return [(arr[i], i) for i in indices]


def generate_n_random_numbers(n):
    return [random.randint(0, 255) for _ in range(n)]



def change_dct(dct, ms, randoms, alpha):
    for i in range(len(ms)):
        index = ms[i][1]
        dct[index] += randoms[i] * alpha


def cox_watermarking(image, output, K, alpha, logfile='cox.log'):
    _, img = open_image(image)
    dct = twoDimensionalDCT(img)
    most_significant = n_max(dct, K)
    randoms = generate_n_random_numbers(K)
    change_dct(dct, most_significant, randoms, alpha)
    idct = inverseTwoDimensionalDCT(dct)
    idct = idct.clip(0, 255)
    idct = idct.astype('uint8')
    img = Image.fromarray(idct)
    img.save(output)

    # write log
    with open(logfile, 'w') as f:
        lines = [str(alpha), str(randoms), str(most_significant)]
        s = [line + '\n' for line in lines]
        f.writelines(s)


def cox_get_watermark(watermarked, logfile='cox.log'):
    lines = []
    with open(logfile, 'r') as f:
        lines = f.readlines()
    alpha = float(lines[0])
    randoms = eval(lines[1])
    gotten_randoms = []
    ms = eval(lines[2])
    _, img = open_image(watermarked)
    dct = twoDimensionalDCT(img)

    for i in range(len(ms)):
        index = ms[i][1]
        previous = ms[i][0]
        gotten_randoms.append((dct[index] - previous) / alpha)

    randoms = numpy.asarray(randoms)
    gotten_randoms = numpy.asarray(gotten_randoms)

    return stats.pearsonr(randoms, gotten_randoms)