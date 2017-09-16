#! /usr/bin/env python3

from network import Network
import numpy as np
import copy
from fixed_trainer import FixedTrainer
import random

if __name__ == "__main__":
    i = [[random.uniform(0,0.5), random.uniform(0,0.5)] for x in range(0, 10)]
    r = [a + b for a, b in i]
    geneticXor = FixedTrainer(20, np.array(i),  np.array(r), [2, 5, 1])

    for i in range(0, 1000):
        error = geneticXor.nextGeneration(1, 0.5)
        print("Error[{}]: {}".format(geneticXor.getGeneration(), error))
        if error < 1e-4:
            break

    print(geneticXor.nets[0])
    i = [[random.uniform(0,0.5), random.uniform(0,0.5)] for x in range(0, 10)]
    for a, b in i:
        print("{} + {} = {} ({})".format(a, b, geneticXor.nets[0].apply(np.array([a, b])), a+b))
