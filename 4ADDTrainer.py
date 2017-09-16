#! /usr/bin/env python3

from network import Network
import numpy as np
import copy
from fixed_trainer import FixedTrainer
import random

def bin_array(n):
    return [int(x) for x in (bin(n)[2:]).zfill(4)]

if __name__ == "__main__":
    i = [[random.randint(0,15), random.randint(0,15)] for x in range(0, 100)]
    r = [(a + b) % 16 for a, b in i]
    geneticXor = FixedTrainer(3, np.array([bin_array(a) + bin_array(b) for a, b in i]),  np.array([bin_array(a) for a in r]), [8, 10, 4])

    while True:
        error = geneticXor.nextGeneration(0.05, 0)
        print("Error[{}]: {}".format(geneticXor.getGeneration(), error))
        if error < 2:
            break

    print(geneticXor.nets[0])
    i = [[random.randint(0,15), random.randint(0,15)] for x in range(0, 30)]
    r = [(a + b) % 16 for a, b in i]
    for a, b in i:
        output = geneticXor.nets[0].apply(np.array(bin_array(a) + bin_array(b)))
        output = ['0' if x < 0.5 else '1' for x in output]
        output = int(''.join(output), 2)
        print("{} + {} = {} ({})".format(a, b, output, (a + b) % 16))
