from random import random, randrange
import pylab

class GeneticAlgorithm(object):
    """
    A genetic algorithm is a model of biological evolution.  It
    maintains a population of chromosomes.  Each chromosome is
    represented as a list of 0's and 1's.  A fitness function must be
    defined to score each chromosome.  Initially, a random population
    is created. Then a series of generations are executed.  Each
    generation, parents are selected from the population based on
    their fitness.  More highly fit chromosomes are more likely to be
    selected to create children.  With some probability crossover will
    be done to model sexual reproduction.  With some very small
    probability mutations will occur.  A generation is complete once
    all of the original parents have been replaced by children.  This
    process continues until the maximum generation is reached or when
    the isDone method returns True.
    """
    def __init__(self, length, popSize, verbose=False):
        self.verbose = verbose      # Set to True to see more info displayed
        self.length = length        # Length of the chromosome
        self.popSize = popSize      # Size of the population
        self.bestEver = None        # Best member ever in this evolution
        self.bestEverScore = 0      # Fitness of best member ever
        self.population = None      # Population is a list of chromosomes
        self.scores = None          # List of fitness of all members of pop
        self.totalFitness = None    # Total fitness in entire population
        self.pCrossover = 0.6       # Probability of crossover
        self.pMutation = 0.1        # Probability of mutation (per bit)
        self.bestList = []          # List of best fitness per generation
        self.avgList = []           # List of avg fitness per generation
        print("Executing genetic algorithm")
        print("Chromosome length:", self.length)
        print("Population size:", self.popSize)

    def initializePopulation(self):
        """
        Initialize each chromosome in the population with a random
        series of 1's and 0's.

        Returns: None
        Result: Initializes self.population
        """
        popn = []
        for i in range(self.popSize):
            chrom = []
            chrom_length = self.length
            for j in range(chrom_length):
                newnum = randrange(0,2,1)
                chrom.append(newnum)
            popn.append(chrom)
        self.population = popn


    def evaluatePopulation(self):
        """
        Computes the fitness of every chromosome in population.  Saves the
        fitness values to the list self.scores.  Checks whether the
        best fitness in the current population is better than
        self.bestEverScore. If so, prints a message that a new best
        was found and its score, updates this variable and saves the
        chromosome to self.bestEver.  Computes the total fitness of
        the population and saves it in self.totalFitness. Appends the
        current bestEverScore to the self.bestList, and the current
        average score of the population to the self.avgList.

        Returns: None
        """
        #raise NotImplementedError("TODO")
        tempscores = []
        best_fitness = 0
        best_chrom = None
        total_fitness = 0
        for chromosome in self.population:
            fitness = self.fitness(chromosome)
            tempscores.append(fitness)
            total_fitness += fitness
            if fitness > best_fitness:
                best_fitness = fitness
                best_chrom = chromosome
        if best_fitness > self.bestEverScore:
            self.bestEverScore = best_fitness
            print("The new best score is:",fitness)
            self.bestEver = best_chrom
        self.bestList.append(self.bestEverScore)
        self.totalFitness = total_fitness
        av = total_fitness/(self.popSize)
        self.avgList.append(av)
        self.scores = tempscores


    def selection(self):
        """
        Each chromosome's chance of being selected for reproduction is
        based on its fitness.  The higher the fitness the more likely
        it will be selected.  Uses the roulette wheel strategy on
        self.scores.

        Returns: A COPY of the selected chromosome. You can make a copy
        of a python list by taking a full slice of it.  For example
        x = [1, 2, 3, 4]
        y = x[:] # y is a copy of x
        """
        spin = random() * self.totalFitness
        partialFitness = 0
        for i in range(self.popSize):
            partialFitness += self.scores[i]
            if partialFitness >= spin:
                break
        return self.population[i][:]

    def crossover(self, parent1, parent2):
        """
        With probability self.pCrossover, recombine the genetic
        material of the given parents at a random location between
        1 and the length-1 of the chromosomes. If no crossover is
        performed, then return the original parents.

        When self.verbose is True, and crossover is done, prints
        the crossover point, and the two children.  Otherwise prints
        that no crossover was done.

        Returns: Two children
        """
        # raise NotImplementedError("TODO")
        if random() <= self.pCrossover:
            # randomly choose location, excluding beginning and end of chrom
            locus = randrange(1, self.length-1)
            # recombine
            child1 = parent1
            child2 = parent2
            temp = child1[locus]
            child1[locus] = child2[locus]
            child2[locus] = temp
            if (self.verbose == True):
                print(f"crossover point: {locus}     child1: {child1}      child2:{child2}")
            return child1, child2
        else:
            # no crossover
            print("No crossover was done.")
            return parent1, parent2

    def mutation(self, chromosome):
        """
        With probability self.pMutation tested at each position in the
        chromosome, flip value.

        When self.verbose is True, if mutation is done prints the
        position of the string being mutated. When no mutations are
        done prints this at the end.

        Returns: None
        """
        # raise NotImplementedError("TODO")
        isMutated = False
        for i in range(len(chromosome)):
            if random() <= self.pMutation:
                isMutated = True
                if chromosome[i] == 0:
                    chromosome[i] = 1
                else:
                    chromosome[i] = 0
                if self.verbose:
                    print(f"mutation at location {i}")
        if not isMutated and self.verbose:
            print("no mutations.")


    def oneGeneration(self):
        """
        Execute one generation of the evolution. Each generation,
        repeatedly select two parents, call crossover to generate two
        children.  Call mutate on each child.  Finally add both
        children to the new population.  Continue until the new
        population is full. Replaces self.pop with a new population.

        When self.verbose is True, prints the parents that were
        selected and their children after crossover and mutation
        have been completed.

        Returns: None
        """
        newpop = []
        while len(newpop)<self.popSize:
           p1 = self.selection()
           p2 = self.selection()
           c1, c2 = self.crossover(p1,p2)
           self.mutation(c1)
           self.mutation(c2)
           newpop.append(c1)
           newpop.append(c2)
           if self.verbose:
               print(f"P1: {p1}, P2: {p2}")
               print(f"C1: {c1}, C2: {c2}")
        if len(newpop) > self.popSize:
            newpop = newpop[:-1]
        self.population = newpop


    def evolve(self, maxGen, pCrossover=0.7, pMutation=0.001):
        """
        Stores the probabilites in appropriate class variables.

        Runs a series of generations until maxGen is reached or
        self.isDone() returns True.

        Returns the best chromosome ever found over the course of
        the evolution, which is stored in self.bestEver.
        """
        self.pCrossover = pCrossover
        self.pMutation = pMutation
        self.initializePopulation()
        self.evaluatePopulation()
        for i in range(maxGen):
            self.oneGeneration()
            self.evaluatePopulation()
            if self.isDone():
                break
        return self.bestEver

    def plotStats(self, title=""):
        """
        Plots a summary of the GA's progress over the generations.
        Adds the given title to the plot.
        """
        gens = range(len(self.bestList))
        pylab.plot(gens, self.bestList, label="Best")
        pylab.plot(gens, self.avgList, label="Average")
        pylab.legend(loc="upper left")
        pylab.xlabel("Generations")
        pylab.ylabel("Fitness")
        if len(title) != 0:
            pylab.title(title)
        pylab.savefig("FitnessByGeneration.png")
        pylab.show()

    def fitness(self, chromosome):
        """
        The fitness function will change for each problem.  Therefore
        it is not defined here.  To use this class to solve a
        particular problem, inherit from this class and define this
        method.
        """
        # Do not implement, this will be overridden
        pass

    def isDone(self):
        """
        The stopping critera will change for each problem.  Therefore
        it is not defined here.  To use this class to solve a
        particular problem, inherit from this class and define this
        method.
        """
        # Do not implement, this will be overridden
        pass
