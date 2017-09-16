#! /usr/bin/env python3

import random
import numpy as np
import math

class Layer:
    def __init__(self, in_c, node_c):
        # print("{}, {}".format(in_c, node_c))
        self.m = np.random.rand(node_c, in_c)
        self.m = self.m / self.m.sum(axis=1)[:,None]
        # print("Matrix: {}".format(self.m))
        self.sigmoid = np.vectorize(lambda x : 1 / (1 + math.exp(-x)))

    def apply(self, input):
        # print("Matrix: {}\nInput: {}".format(self.m, input))
        return self.sigmoid(self.m.dot(input))

    def __str__(self):
         return "{}".format(self.m)

    def __repr__(self):
        return "{}".format(self.m)

class Network:
    def __init__(self, layers):
        self.layers = []
        last = layers[0]
        for x in layers[1:]:
            self.layers += [Layer(last, x)]
            last = x
        # print(self.layers)
        # print()

    def apply(self, input):
        for l in self.layers:
            input = l.apply(input)
        return input

if __name__ == '__main__':
    n = Network([16, 2, 5])
    print(n.apply(np.random.rand(16)))
