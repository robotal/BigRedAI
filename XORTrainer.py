#! /usr/bin/env python3

from network import Network
import numpy as np
import copy
from fixed_trainer import FixedTrainer

if __name__ == "__main__":
    geneticXor = FixedTrainer(np.array([[1, 0], [0, 1], [1, 1], [0, 0]]),  np.array([[1], [1], [0], [0]]), [2, 2, 1])

    for i in range(0, 1000):
        geneticXor.nextGeneration(0.3, 0.05)

    print(geneticXor.nets[0])
    print(geneticXor.nets[0].apply(np.array([0, 0])))
    print(geneticXor.nets[0].apply(np.array([1, 0])))
    print(geneticXor.nets[0].apply(np.array([0, 1])))
    print(geneticXor.nets[0].apply(np.array([1, 1])))
