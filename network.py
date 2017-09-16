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

    def mutate(self, sigma):
        self.m += np.random.normal(loc = 0.0, scale = sigma, size=self.m.shape)

    def __str__(self):
         return "{}".format(self.m)

    def __repr__(self):
        return "{}".format(self.m)

class Network:
    def __init__(self, layers):
        self.layers = []
        last = layers[0]
        for x in layers[1:]:
            self.layers.append(Layer(last, x))
            last = x
        # print(self.layers)
        # print()

    def apply(self, input):
        for l in self.layers:
            input = l.apply(input)
        return input

    def mutate(self, sigma):
        for l in self.layers:
            l.mutate(sigma)



if __name__ == '__main__':
    n = Network([16, 2, 5])
    foo = np.random.rand(16)
    for i in range(0,10000):
        print(n.apply(foo))
        n.mutate(1)
