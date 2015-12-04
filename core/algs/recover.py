from abstract import Algorithm
from PIL import Image
import random
from bitstring import BitArray
import numpy as np

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
        r_matrix, g_matrix, b_matrix = self.divide_in_blocks(r,g,b, self.NUM)

        # Randomly pick a prime number k E [1, N - 1]
        k = self.get_k(self.N)

        name = image_file.split('.')[0]

        with open(Algorithm().get_image_output_file('{0}_k'.split('.')[0].format(name)), 'w+') as fd:
            fd.write(str(k))

        # For each block number X, apply Eq. (3) to obtain X' the number of its mapping block.
        # Record all pairs of X and X' to form the block mapping sequence
        cor = {}
        
        for v in r_matrix:
            cor[v] = (k*(v+1))%self.N#((k*(v+1))%self.N)+1
        # Block watermark embedding

        new_r_matrix = {}
        new_g_matrix = {}
        new_b_matrix = {}


        for k in r_matrix:
            new_r_matrix[k],new_g_matrix[k],new_b_matrix[k] = \
                self.divide_in_blocks(r_matrix[k], g_matrix[k], b_matrix[k], 2)



        # Sub-block watermark generation and embedding algorithm
        # Set the two LSBs of each pixel within the block to zero

        for k in new_r_matrix:
            for j in new_r_matrix[k]:
                for x in range(len(new_r_matrix[k][j])):
                    for y in range(len(new_r_matrix[k][j][0])):
                        new_r_matrix[k][j][(x,y)] = self.change_lsb(new_r_matrix[k][j][(x,y)], 2, 0)
                        new_g_matrix[k][j][(x,y)] = self.change_lsb(new_g_matrix[k][j][(x,y)], 2, 0)
                        new_b_matrix[k][j][(x,y)] = self.change_lsb(new_b_matrix[k][j][(x,y)], 2, 0)


        # Compute the average intensity of the block and each
        # of its four sub-blocks, denoted by avg_{r,g,b}(calculated before) and avg_{r,g,b}' ,respectively.
        # Generate the authentication watermark v of each sub-block
        # Generate the parity-check bit p of each sub-block

        # From the mapping sequence generated in the preparation step, obtain block A whose recovery
        # information will be storedin block B. {cor}

        # Compute the average intensity of each corresponding sub-block A

        # point 6, 7
        r_avg, g_avg, b_avg = {}, {}, {}
        for Bk in new_r_matrix:
            r_avg[Bk] = {}
            r_avg[Bk]['avg'] = self.avg_int(r_matrix[Bk])
            g_avg[Bk] = {}
            g_avg[Bk]['avg'] = self.avg_int(g_matrix[Bk])
            b_avg[Bk] = {}
            b_avg[Bk]['avg'] = self.avg_int(b_matrix[Bk])
            for Bs in new_r_matrix[Bk]:
                r_avg[Bk][Bs] = self.avg_int(new_r_matrix[Bk][Bs])
                g_avg[Bk][Bs] = self.avg_int(new_g_matrix[Bk][Bs])
                b_avg[Bk][Bs] = self.avg_int(new_b_matrix[Bk][Bs])

        r_v, g_v, b_v = {}, {}, {}
        for Bk in new_r_matrix:
            r_v[Bk] = {}
            g_v[Bk] = {}
            b_v[Bk] = {}
            for Bs in new_r_matrix[Bk]:
                r_v[Bk][Bs] = 1 if r_avg[Bk][Bs] >= r_avg[Bk]['avg'] else 0
                g_v[Bk][Bs] = 1 if g_avg[Bk][Bs] >= g_avg[Bk]['avg'] else 0
                b_v[Bk][Bs] = 1 if b_avg[Bk][Bs] >= b_avg[Bk]['avg'] else 0

        r_p, g_p, b_p = {}, {}, {}
        for Bk in new_r_matrix:
            r_p[Bk] = {}
            g_p[Bk] = {}
            b_p[Bk] = {}
            for Bs in new_r_matrix[Bk]:
                num = self.count_bit_msb(r_avg[Bk][Bs], 6, 1)
                r_p[Bk][Bs] = 1 if num%2 == 1 else 0
                num = self.count_bit_msb(g_avg[Bk][Bs], 6, 1)
                g_p[Bk][Bs] = 1 if num%2 == 1 else 0
                num = self.count_bit_msb(b_avg[Bk][Bs], 6, 1)
                b_p[Bk][Bs] = 1 if num%2 == 1 else 0


        A = cor[0]
        B = cor[A]
        while True:
            for As in new_r_matrix[A]:
                r_avg_A = r_avg[A][As]
                g_avg_A = g_avg[A][As]
                b_avg_A = b_avg[A][As]

                r_avg_A = self.truncate_x_lsb(r_avg_A, 2)
                g_avg_A = self.truncate_x_lsb(g_avg_A, 2)
                b_avg_A = self.truncate_x_lsb(b_avg_A, 2)

                self.embed_matrix_lsb(new_r_matrix[B][As], r_v[B][As], r_p[B][As], r_avg_A)
                self.embed_matrix_lsb(new_g_matrix[B][As], g_v[B][As], g_p[B][As], g_avg_A)
                self.embed_matrix_lsb(new_b_matrix[B][As], b_v[B][As], b_p[B][As], b_avg_A)

            if B == cor[0]:
                break
            A = B
            B = cor[A]

        """print(cor)

        i = 0
        k = -1
        Ak = i
        Bk = cor[i]
        while True:
            #Ak = AAk-1
            #Bk = BBk-1
            print(Ak,Bk)
            k +=1
            new_r_matrix[Bk]["avg"] = {"avg":self.avg_int(r_matrix[Bk])}
            new_g_matrix[Bk]["avg"] = {"avg":self.avg_int(g_matrix[Bk])}
            new_b_matrix[Bk]["avg"] = {"avg":self.avg_int(b_matrix[Bk])}
            new_r_matrix[Ak]["avg"] = {"avg":self.avg_int(r_matrix[Ak])}
            new_g_matrix[Ak]["avg"] = {"avg":self.avg_int(g_matrix[Ak])}
            new_b_matrix[Ak]["avg"] = {"avg":self.avg_int(b_matrix[Ak])}
            new_r_matrix[Bk]["3-tuple"] = {}
            new_g_matrix[Bk]["3-tuple"] = {}
            new_b_matrix[Bk]["3-tuple"] = {}
            for j in new_r_matrix[Bk]:
                if j != "avg" and j != "3-tuple":
                    new_r_matrix[Ak]["avg"][j] = self.avg_int(new_r_matrix[Ak][j])
                    new_g_matrix[Ak]["avg"][j] = self.avg_int(new_g_matrix[Ak][j])
                    new_b_matrix[Ak]["avg"][j] = self.avg_int(new_b_matrix[Ak][j])

                    r_truncated_A = self.truncate_x_lsb(new_r_matrix[Ak]["avg"][j], 2)
                    g_truncated_A = self.truncate_x_lsb(new_g_matrix[Ak]["avg"][j], 2)
                    b_truncated_A = self.truncate_x_lsb(new_b_matrix[Ak]["avg"][j], 2)

                    new_r_matrix[Bk]["avg"][j] = self.avg_int(new_r_matrix[Bk][j])
                    new_r_matrix[Bk]["3-tuple"][j] = {}
                    new_r_matrix[Bk]["3-tuple"][j]["v"] = \
                        1 if new_r_matrix[Bk]["avg"][j] >= new_r_matrix[Bk]["avg"]["avg"] else 0
                    new_r_matrix[Bk]["3-tuple"][j]["p"] = self.count_bit_msb(new_r_matrix[Bk]["avg"][j], 6, 1) % 2
                    self.embed_matrix_lsb(
                        new_r_matrix[Bk][j],
                        new_r_matrix[Bk]["3-tuple"][j]["v"],
                        new_r_matrix[Bk]["3-tuple"][j]["p"],
                        r_truncated_A
                    )
                    new_g_matrix[Bk]["avg"][j] = self.avg_int(new_g_matrix[Bk][j])
                    new_g_matrix[Bk]["3-tuple"][j] = {}
                    new_g_matrix[Bk]["3-tuple"][j]["v"] = \
                        1 if new_g_matrix[Bk]["avg"][j] >= new_g_matrix[Bk]["avg"]["avg"] else 0
                    new_g_matrix[Bk]["3-tuple"][j]["p"] = self.count_bit_msb(new_g_matrix[Bk]["avg"][j], 6, 1) % 2
                    self.embed_matrix_lsb(
                        new_g_matrix[Bk][j],
                        new_g_matrix[Bk]["3-tuple"][j]["v"],
                        new_g_matrix[Bk]["3-tuple"][j]["p"],
                        g_truncated_A
                    )
                    new_b_matrix[Bk]["avg"][j] = self.avg_int(new_b_matrix[Bk][j])
                    new_b_matrix[Bk]["3-tuple"][j] = {}
                    new_b_matrix[Bk]["3-tuple"][j]["v"] = \
                        1 if new_b_matrix[Bk]["avg"][j] >= new_b_matrix[Bk]["avg"]["avg"] else 0
                    new_b_matrix[Bk]["3-tuple"][j]["p"] = self.count_bit_msb(new_b_matrix[Bk]["avg"][j], 6, 1) % 2
                    self.embed_matrix_lsb(
                        new_b_matrix[Bk][j],
                        new_b_matrix[Bk]["3-tuple"][j]["v"],
                        new_b_matrix[Bk]["3-tuple"][j]["p"],
                        b_truncated_A
                    )
            if Bk == i:
                break
            Ak = Bk
            Bk = cor[Ak]"""

        return image

    def embed_matrix_lsb(self, matrix, v, p, r):
        vpr = str(v) + str(p) + str(r)

        for x in range(len(matrix)):
            for y in range(len(matrix[0])):
                self.embed_pixel_lsb(matrix, (x,y) ,vpr[y+(x*len(matrix))]+vpr[y+(x*len(matrix))+4])


    def embed_pixel_lsb(self, matrix, i, to_emb):
        snum = "{0:b}".format(matrix[i])
        if len(snum) <= len(to_emb):
            b = BitArray(bin=to_emb)
            matrix[i] = b.uint
            return
        snum = snum[:len(to_emb)*-1] + to_emb
        b = BitArray(bin=snum)
        matrix[i] = b.uint


    def truncate_x_lsb(self, number, x):
        snum = "{0:b}".format(number)
        if len(snum) < 8:
            missing = 8 - len(snum)
            snum = "0" * missing + snum

        return snum[:-1*x]

    def count_bit_msb(self, number, num_of_bits, n_val):
        snum = "{0:b}".format(number)
        if num_of_bits>8:
            num_of_bits = 8
        if len(snum) < 8:
            snum = (8 - len(snum))*"0"+snum
        count = 0
        for i in range(0, num_of_bits):
            count += 1 if snum[i] == str(n_val) else 0
        return count

    def avg_int(self, matrix):
        return int(np.matrix(matrix).mean())

    def change_lsb(self, number, num_of_bits, n_val):
        snum = "{0:b}".format(number)
        if len(snum) <= num_of_bits:
            b = BitArray(bin=str(n_val)*num_of_bits)
            return b.uint
        snum = snum[:(num_of_bits)*-1] + str(n_val)*num_of_bits
        b = BitArray(bin=snum)
        return b.uint


    def is_prime(self, a):
        return all(a % i for i in xrange(2, a))

    def get_k(self,n):
        primes = [i for i in range(1,n) if self.is_prime(i)]
        n = random.choice(primes)
        return n

    def divide_in_blocks(self, r, g, b, num):
        r_matrix = {}
        g_matrix = {}
        b_matrix = {}
        k = num
        z = num
        x = 0
        y = 0
        self.N = (len(r)/num)**2
        num = len(r)/num
        for i in range(k, len(r)+1, k):
            for j in range(z, len(r[0])+1, z):
                r_matrix[y+(x*num)] = r[i-k:i, j-z:j]
                g_matrix[y+(x*num)] = g[i-k:i, j-z:j]
                b_matrix[y+(x*num)] = b[i-k:i, j-z:j]
                y += 1
            x += 1
            y = 0
        return r_matrix, g_matrix, b_matrix

    def extract_specific(self, image, watermark):
        import numpy
        erroneous, _, _ = self.tamper_detection_level_1(numpy.array(image), watermark)

        r,g,b = self.split_image(image)

        # Divide the image into non-overlapping blocks of 4X4 pixels
        blocks = self.divide_in_blocks(r,g,b, self.NUM)

        new_r_matrix = {}
        new_g_matrix = {}
        new_b_matrix = {}
        for k in blocks[0]:
            new_r_matrix[k],new_g_matrix[k],new_b_matrix[k] = \
                self.divide_in_blocks(blocks[0][k], blocks[1][k], blocks[2][k], 2)

        sblocks = [new_r_matrix, new_g_matrix, new_b_matrix]

        r_erroneous = erroneous[0]
        g_erroneous = erroneous[1]
        b_erroneous = erroneous[2]

        erroneous_blocks = set()
        for rblock in r_erroneous:
            if r_erroneous[rblock] is True:
                erroneous_blocks.add(rblock[0])
        for gblock in g_erroneous:
            if g_erroneous[gblock] is True:
                erroneous_blocks.add(gblock[0])
        for bblock in b_erroneous:
            if b_erroneous[bblock] is True:
                erroneous_blocks.add(bblock[0])

        k = -1
        with open(Algorithm().get_image_output_file(watermark)) as fd:
            k = int(fd.read())

        for block in erroneous_blocks:
            block_number = block
            r_invalid_block = sblocks[0][block_number]
            g_invalid_block = sblocks[1][block_number]
            b_invalid_block = sblocks[2][block_number]

            next_block = (k*(block_number+1)) % len(blocks[0])


            if next_block in erroneous_blocks:
                continue

            r_next_block = sblocks[0][next_block]
            g_next_block = sblocks[1][next_block]
            b_next_block = sblocks[2][next_block]

            r_avg = {}
            for rBs in r_next_block:
                binary_avg = ""
                subblock = r_next_block[rBs]

                binary_avg += str(self.get_bit_value(subblock[(1,0)], 1))
                binary_avg += str(self.get_bit_value(subblock[(1,1)], 1))
                binary_avg += str(self.get_bit_value(subblock[(0,0)], 0))
                binary_avg += str(self.get_bit_value(subblock[(0,1)], 0))
                binary_avg += str(self.get_bit_value(subblock[(1,0)], 0))
                binary_avg += str(self.get_bit_value(subblock[(1,1)], 0))
                binary_avg += "00"
                r_avg[rBs] = BitArray(bin=binary_avg).uint

            g_avg = {}
            for rGs in g_next_block:
                binary_avg = ""
                subblock = g_next_block[rGs]

                binary_avg += str(self.get_bit_value(subblock[(1,0)], 1))
                binary_avg += str(self.get_bit_value(subblock[(1,1)], 1))
                binary_avg += str(self.get_bit_value(subblock[(0,0)], 0))
                binary_avg += str(self.get_bit_value(subblock[(0,1)], 0))
                binary_avg += str(self.get_bit_value(subblock[(1,0)], 0))
                binary_avg += str(self.get_bit_value(subblock[(1,1)], 0))
                binary_avg += "00"
                g_avg[rGs] = BitArray(bin=binary_avg).uint

            b_avg = {}
            for rBs in b_next_block:
                binary_avg = ""
                subblock = b_next_block[rBs]
                binary_avg += str(self.get_bit_value(subblock[(1,0)], 1))
                binary_avg += str(self.get_bit_value(subblock[(1,1)], 1))
                binary_avg += str(self.get_bit_value(subblock[(0,0)], 0))
                binary_avg += str(self.get_bit_value(subblock[(0,1)], 0))
                binary_avg += str(self.get_bit_value(subblock[(1,0)], 0))
                binary_avg += str(self.get_bit_value(subblock[(1,1)], 0))
                binary_avg += "00"
                b_avg[rBs] = BitArray(bin=binary_avg).uint


            avg = [r_avg, g_avg, b_avg]
            invblock = [r_invalid_block, g_invalid_block, b_invalid_block]

            for color in range(3):
                color_avg = avg[color]
                color_invalid_block = invblock[color]
                for bb in color_invalid_block:
                    subblock = color_invalid_block[bb]
                    for x in range(2):
                        for y in range(2):
                            subblock[(x,y)] = color_avg[bb]

        return image, 0

    def get_watermark_name(self, filename):
        import os
        _, filename = os.path.split(filename)
        return self.get_image_output_file(_+"recovered-"+filename.split('.')[0]+".png")


    def get_bit_value(self, value, bit):
        snum = "{0:b}".format(value)
        if len(snum) <= bit:
            return 0
        else:
            return int(snum[-1-bit])


    def tamper_detection_level_1(self, image, image_file):


        r,g,b = self.split_image(image)

        # Divide the image into non-overlapping blocks of 4X4 pixels
        r_matrix, g_matrix, b_matrix = self.divide_in_blocks(r,g,b, self.NUM)

        # Block watermark embedding

        new_r_matrix = {}
        new_g_matrix = {}
        new_b_matrix = {}


        for k in r_matrix:
            new_r_matrix[k],new_g_matrix[k],new_b_matrix[k] = \
                self.divide_in_blocks(r_matrix[k], g_matrix[k], b_matrix[k], 2)

        r_aux, g_aux, b_aux = {},{},{}
        p_r, v_r, p_g, v_g, p_b, v_b = {}, {}, {}, {}, {}, {}
        for a in new_r_matrix:
            for b in new_r_matrix[a]:
                v_r[(a, b)] = self.get_bit_value(new_r_matrix[a][b][(0, 0)], 1)
                p_r[(a, b)] = self.get_bit_value(new_r_matrix[a][b][(0, 1)], 1)
                v_g[(a, b)] = self.get_bit_value(new_g_matrix[a][b][(0, 0)], 1)
                p_g[(a, b)] = self.get_bit_value(new_g_matrix[a][b][(0, 1)], 1)
                v_b[(a, b)] = self.get_bit_value(new_b_matrix[a][b][(0, 0)], 1)
                p_b[(a, b)] = self.get_bit_value(new_b_matrix[a][b][(0, 1)], 1)
                r_aux[(a, b)] = False
                g_aux[(a, b)] = False
                b_aux[(a, b)] = False


        # Sub-block watermark generation and embedding algorithm
        # Set the two LSBs of each pixel within the block to zero

        for k in new_r_matrix:
            for j in new_r_matrix[k]:
                for x in range(len(new_r_matrix[k][j])):
                    for y in range(len(new_r_matrix[k][j][0])):
                        new_r_matrix[k][j][(x,y)] = self.change_lsb(new_r_matrix[k][j][(x,y)], 2, 0)
                        new_g_matrix[k][j][(x,y)] = self.change_lsb(new_g_matrix[k][j][(x,y)], 2, 0)
                        new_b_matrix[k][j][(x,y)] = self.change_lsb(new_b_matrix[k][j][(x,y)], 2, 0)


        # Compute the average intensity of the block and each
        # of its four sub-blocks, denoted by avg_{r,g,b}(calculated before) and avg_{r,g,b}' ,respectively.
        # Generate the authentication watermark v of each sub-block
        # Generate the parity-check bit p of each sub-block

        # From the mapping sequence generated in the preparation step, obtain block A whose recovery
        # information will be storedin block B. {cor}

        # Compute the average intensity of each corresponding sub-block A

        # point 6, 7
        r_avg, g_avg, b_avg = {}, {}, {}
        for Bk in new_r_matrix:
            r_avg[Bk] = {}
            r_avg[Bk]['avg'] = self.avg_int(r_matrix[Bk])
            g_avg[Bk] = {}
            g_avg[Bk]['avg'] = self.avg_int(g_matrix[Bk])
            b_avg[Bk] = {}
            b_avg[Bk]['avg'] = self.avg_int(b_matrix[Bk])
            for Bs in new_r_matrix[Bk]:
                r_avg[Bk][Bs] = self.avg_int(new_r_matrix[Bk][Bs])
                g_avg[Bk][Bs] = self.avg_int(new_g_matrix[Bk][Bs])
                b_avg[Bk][Bs] = self.avg_int(new_b_matrix[Bk][Bs])

        r_v, g_v, b_v = {}, {}, {}
        for Bk in new_r_matrix:
            r_v[Bk] = {}
            g_v[Bk] = {}
            b_v[Bk] = {}
            for Bs in new_r_matrix[Bk]:
                r_v[Bk][Bs] = 1 if r_avg[Bk][Bs] >= r_avg[Bk]['avg'] else 0
                g_v[Bk][Bs] = 1 if g_avg[Bk][Bs] >= g_avg[Bk]['avg'] else 0
                b_v[Bk][Bs] = 1 if b_avg[Bk][Bs] >= b_avg[Bk]['avg'] else 0

        r_p, g_p, b_p = {}, {}, {}
        for Bk in new_r_matrix:
            r_p[Bk] = {}
            g_p[Bk] = {}
            b_p[Bk] = {}
            for Bs in new_r_matrix[Bk]:
                num = self.count_bit_msb(r_avg[Bk][Bs], 6, 1)
                r_p[Bk][Bs] = 1 if num%2 == 1 else 0
                num = self.count_bit_msb(g_avg[Bk][Bs], 6, 1)
                g_p[Bk][Bs] = 1 if num%2 == 1 else 0
                num = self.count_bit_msb(b_avg[Bk][Bs], 6, 1)
                b_p[Bk][Bs] = 1 if num%2 == 1 else 0


        for k in p_r:
            if p_r[k] != r_p[k[0]][k[1]]:
                r_aux[k] = True
            if p_g[k] != g_p[k[0]][k[1]]:
                g_aux[k] = True
            if p_b[k] != b_p[k[0]][k[1]]:
                b_aux[k] = True

        for k in v_r:
            if v_r[k] != r_v[k[0]][k[1]]:
                r_aux[k] = True
            if v_g[k] != g_v[k[0]][k[1]]:
                g_aux[k] = True
            if v_b[k] != b_v[k[0]][k[1]]:
                b_aux[k] = True

        return [r_aux, g_aux, b_aux], [r_matrix, g_matrix, b_matrix], [new_r_matrix, new_b_matrix, new_g_matrix]

