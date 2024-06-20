from ga import *
from neural_net import *
from random import random

class NeuralGA(GeneticAlgorithm):
    """
    A specialized version of the GA that evolves the weights and biases
    of a fixed neural network architecture.
    """
    def __init__(self, in_size, out_size, hid_sizes, pop_size):
        """
        Must specify the sizes of all of the layers within the network,
        as well as the size of the GA population.
        """
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes

        # Create a dummy neural net to see how many weights it contains
        nn = Network(in_size, out_size, hid_sizes)
        length = len(nn.getWeights())

        # Call the parent class constructor
        super(NeuralGA, self).__init__(length, pop_size)

    def randomWeight(self, magnitude=0.15):
        """
        Generate a random weight that has an absolute value less than
        the given magnitude, and is equally likely to be positive or
        negative.
        """
        val = random() * magnitude
        if random() < 0.5:
            val *= -1
        return val

    def initializePopulation(self):
        """
        Override the GA's method of initialization.
        Initialize each chromosome in self.population with random weights.

        Returns: None
        """
        popn = []
        for i in range(self.popSize):
            chrom = []
            chrom_length = self.length
            for j in range(chrom_length):
                chrom.append(self.randomWeight())
            popn.append(chrom)
        self.population = popn

    def mutation(self, chromosome):
        """
        Override the GA's method of mutation.
        With probability self.pMutation, mutate positions in the
        chromosome by adding the amount of a random weight.
        """
        for bit in chromosome:
            if random() <= self.pMutation:
                bit += self.randomWeight()
        else:
            if self.verbose:
                print("no mutation")


if __name__ == '__main__':
    # Test the overridden method initializePopulation and mutation
    ga = NeuralGA(2, 1, [], 5)
    ga.initializePopulation()
    ga.pMutation = 0.3 # 30% chance of mutating each weight
    for chromo in ga.population:
        print(chromo)
        print("Calling mutation on this chromosome, result is:")
        ga.mutation(chromo)
        print(chromo)
        print("--------------")
