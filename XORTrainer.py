from network import Network

class XORTrainer:

    def init(self):

        # create a neural net with 2 inputs, 1 output, and a hidden layer
        # with 2 nodes
        net = Network.init([2, 2, 1])
