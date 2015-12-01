from abstract import Algorithm
from PIL import Image
import random

'''
    http://www1.pu.edu.tw/~lan/papers/watermarking.pdf
'''

class Recover(Algorithm):

    NUM = 4

    def split_image(self, image):
        return image[:, :, 2], image[:, :, 1], image[:, :, 0]

    def embed_specific(self, image, image_file, watermark=None):

        r,g,b = self.split_image(image)

        # Divide the image into non-overlapping blocks of 4X4 pixels
        r_matrix, g_matrix, b_matrix = self.divide_in_blocks(r,g,b)

        # Randomly pick a prime number k E [1, N - 1]
        k = self.get_k(self.N)

        # For each block number X, apply Eq. (3) to obtain X' the number of its mapping block.
        # Record all pairs of X and X' to form the block mapping sequence
        cor = {}
        
        for v in r_matrix:
            cor[v] = ((k*v)%self.N)+1


        return image
        pass

    def is_prime(self, a):
        return all(a % i for i in xrange(2, a))

    def get_k(self,n):
        primes = [i for i in range(0,n) if self.is_prime(i)]
        n = random.choice(primes)
        return n

    def divide_in_blocks(self, r, g ,b):
        r_matrix = {}
        g_matrix = {}
        b_matrix = {}
        k = self.NUM
        z = self.NUM
        self.N = (len(r)/self.NUM)**2
        for i in range(k, len(r)+1, k):
            for j in range(z, len(r[0])+1, z):
                r_matrix[j+(i*self.N)] = r[i-k:i, j-z:j]
                g_matrix[j+(i*self.N)] = g[i-k:i, j-z:j]
                b_matrix[j+(i*self.N)] = b[i-k:i, j-z:j]
        return r_matrix, g_matrix, b_matrix

    def extract_specific(self, image, watermark):
        pass