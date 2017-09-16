#! /usr/bin/env python3

import random
import numpy as np
import math


class Layer:
    def __init__(self, in_c, node_c):
        # print("{}, {}".format(in_c, node_c))
        self.m = np.random.rand(node_c, in_c)
        self.m = self.m / self.m.sum(axis=1)[:, None]
        # print("Matrix: {}".format(self.m))
        def safe_sigmoid(x):
            try:
                return 1 / (1 + math.exp(-x))
            except:
                return 0
        self.sigmoid = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))

    def apply(self, input):
        # print("Matrix: {}\nInput: {}".format(self.m, input))
        return self.sigmoid(self.m.dot(input))

    def mutate(self, sigma, mutationProb):
        self.m += np.random.normal(loc=0.0, scale=sigma, size=self.m.shape)

        def randomChange(x):
            if random.uniform(0, 1) < mutationProb:
                return random.uniform(-2, 2) * x
            else:
                return x

        randomlyChange = np.vectorize(randomChange)
        self.m = randomlyChange(self.m)

    def mate(l1, l2):
        r = Layer(1, 1)
        r.m = (l1.m + l2.m) / 2
        return r

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

    def mutate(self, sigma, mutationProb):
        for l in self.layers:
            l.mutate(sigma, mutationProb)

    def mate(n1, n2):
        r = Network([0])
        r.layers = [Layer.mate(a, b) for a, b in zip(n1.layers, n2.layers)]
        return r

    def __str__(self):
        return '; '.join([str(l) for l in self.layers])

    def __repr__(self):
        return str(self)


if __name__ == '__main__':
    n = Network([16, 2, 5])
    foo = np.random.rand(16)
    for i in range(0, 10000):
        print(n.apply(foo))
        n.mutate(1, .05)
