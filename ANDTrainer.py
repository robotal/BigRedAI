#! /usr/bin/env python3

from network import Network
import numpy as np
import copy
from fixed_trainer import FixedTrainer

if __name__ == "__main__":
    geneticXor = FixedTrainer(20, np.array([[1, 0], [0, 1], [1, 1], [0, 0]]),  np.array([[0], [0], [1], [0]]), [2, 10, 1])

    for i in range(0, 1000):
        error = geneticXor.nextGeneration(0.3, 0.05)
        print("Error[{}]: {}".format(geneticXor.getGeneration(), error))
        if error < 1e-5:
            break

    print(geneticXor.nets[0])
    print(geneticXor.nets[0].apply(np.array([0, 0])))
    print(geneticXor.nets[0].apply(np.array([1, 0])))
    print(geneticXor.nets[0].apply(np.array([0, 1])))
    print(geneticXor.nets[0].apply(np.array([1, 1])))
