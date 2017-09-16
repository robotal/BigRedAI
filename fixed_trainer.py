from network import Network
import numpy as np
import copy
import itertools
import multiprocessing as mp

def do_stuff(foo):
    net, ins, outs = foo
    fitness = 0
    for inp, expected in zip(ins, outs):
        # print(net)
        dif = (expected - net.apply(inp))
        fitness += dif.dot(dif)

    return (net, fitness)

class FixedTrainer:

    def __init__(self, n, inputs, expected, layers):

        # create a neural net with 2 inputs, 1 output, and a hidden layer
        # with 2 nodes
        self.nets = []
        self.generation = 0
        self.inputs = inputs
        self.expected = expected
        self.n = n
        self.pool = mp.Pool()

        for i in range(0, 2*(2*n + n*(n-1))):
            self.nets.append(Network(layers))

    def getGeneration(self):
        return self.generation

    def nextGeneration(self, sigma, mutationProb):

        self.generation += 1

        fitnessList = self.pool.map(do_stuff, [(net, self.inputs, self.expected) for net in self.nets])

        fitnessList.sort(key=lambda x: x[1])

        self.nets =  [x[0] for x in fitnessList[0:self.n]]
        self.nets += [Network.mate(a, b) for a, b in itertools.combinations(self.nets, 2)]
        self.nets += [Network.mutate(a, sigma, mutationProb) for a in self.nets]
        self.nets += [Network.mutate(x[0], sigma, mutationProb) for x in fitnessList[0:2*self.n + self.n*(self.n-1)]]

        return fitnessList[0][1]
