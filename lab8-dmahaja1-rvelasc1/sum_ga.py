from ga import *

class SumGA(GeneticAlgorithm):
    """
    An example of using the GeneticAlgorithm class to solve a particular
    problem, in this case finding strings with the maximum number of 1's.
    """
    def fitness(self, chromosome):
        """
        Fitness is the sum of the bits.
        """
        return sum(chromosome)

    def isDone(self):
        """
        Stop when the fitness of the the best member of the current
        population is equal to the maximum fitness.
        """
        return self.fitness(self.bestEver) == self.length


def main():
    # use this main program to incrementally test the GeneticAlgorithm
    # class as you implement it
    """
    ga = SumGA(10, 20, verbose=True)
    ga.initializePopulation()
    ga.evaluatePopulation()
    p1 = ga.selection()
    p2 = ga.selection()
    print(f"p1: {p1}    p2:{p2}")
    c1, c2 = ga.crossover(p1, p2)
    ga.mutation(c1)
    """
    # Chromosomes of length 20, population of size 50
    ga = SumGA(20, 50)
    # Evolve for 100 generations
    # High prob of crossover, low prob of mutation
    bestFound = ga.evolve(100, 0.6, 0.01)
    print(bestFound)
    ga.plotStats("Sum GA")

if __name__ == '__main__':
    main()
