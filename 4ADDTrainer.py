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
    geneticXor = FixedTrainer(3, np.array([bin_array(a) + bin_array(b) for a, b in i]),  np.array([bin_array(a) for a in r]), [8, 20, 4])

    i = [[random.randint(0,15), random.randint(0,15)] for x in range(0, 30)]
    r = [(a + b) % 16 for a, b in i]
    while True:
        error = geneticXor.nextGeneration(0.02, 0.001)
        if geneticXor.getGeneration() & 0x7 == 0:
            print("Error[{}]: {}".format(geneticXor.getGeneration(), error))
        if error < 14:
            break

    print(geneticXor.nets[0])

    for a, b in i:
        output = geneticXor.nets[0].apply(np.array(bin_array(a) + bin_array(b)))
        output = ['0' if x < 0.5 else '1' for x in output]
        output = int(''.join(output), 2)
        print("{} + {} = {} ({}) [{}  {}]".format(str(a).zfill(2), str(b).zfill(2), str(output).zfill(2), str((a + b) % 16).zfill(2), (bin(output)[2:]).zfill(4), (bin((a + b) % 16)[2:]).zfill(4)))
