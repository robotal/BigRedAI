from network import Network
import numpy as np
import copy


class XORTrainer:

    def __init__(self):

        # create a neural net with 2 inputs, 1 output, and a hidden layer
        # with 2 nodes
        self.nets = []
        self.generation = 0
        self.inputs = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        self.expected = np.array([[1], [1], [0], [0]])

        for i in range(0, 12):
            self.nets.append(Network([2, 2, 1]))

    def getGeneration(self):
        return self.generation

    def nextGeneration(self):

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

        print (fitnessList[0][1])

        first = fitnessList[0][0]
        second = fitnessList[1][0]
        third = fitnessList[2][0]
        self.nets = [first, second, third]
        self.nets.append(Network.mate(first, second))
        self.nets.append(Network.mate(third, second))
        self.nets.append(Network.mate(first, third))

        for i in range(0, 6):
            netcopy = copy.deepcopy(self.nets[i])
            netcopy.mutate(.3)
            self.nets.append(netcopy)


if __name__ == "__main__":
    geneticXor = XORTrainer()

    for i in range(0, 1000):
        geneticXor.nextGeneration()

    print(geneticXor.nets[0])
    print(geneticXor.nets[0].apply(np.array([0, 0])))
    print(geneticXor.nets[0].apply(np.array([1, 0])))
    print(geneticXor.nets[0].apply(np.array([0, 1])))
    print(geneticXor.nets[0].apply(np.array([1, 1])))
