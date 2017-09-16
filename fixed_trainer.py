from network import Network
import itertools


class FixedTrainer:

    def __init__(self, n, inputs, expected, layers):

        # create a neural net with 2 inputs, 1 output, and a hidden layer
        # with 2 nodes
        self.nets = []
        self.generation = 0
        self.inputs = inputs
        self.expected = expected
        self.n = n

        for i in range(0, 2*n + n*(n-1)):
            self.nets.append(Network(layers))

    def getGeneration(self):
        return self.generation

    def nextGeneration(self, sigma, mutationProb):

        fitnessList = []
        self.generation += 1

        for net in self.nets:
            fitness = 0
            for inp, expected in zip(self.inputs, self.expected):
                # print(net)
                dif = (expected - net.apply(inp))
                fitness += dif.dot(dif)

            fitnessList.append((net, fitness))

        fitnessList.sort(key=lambda x: x[1])

        self.nets = [x[0] for x in fitnessList[0:self.n]]
        self.nets += [Network.mate(a, b) for a, b in
                      itertools.combinations(self.nets, 2)]
        self.nets += [Network.mutate(a, sigma, mutationProb)
                      for a in self.nets]

        return fitnessList[0][1]
