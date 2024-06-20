########################################
# CS63: Artificial Intelligence, Lab 2
# Fall 2022, Swarthmore College
########################################

from math import exp
from numpy.random import choice, multinomial 

def stochastic_beam_search(problem, pop_size, steps, init_temp,
                           temp_decay, max_neighbors):
        """Implementes the stochastic beam search local search algorithm.
        Inputs:
        - problem: A TSP instance.
        - pop_size: Number of candidates tracked.
        - steps: The number of moves to make in a given run.
        - init_temp: Initial temperature. Note that temperature has a
                slightly different interpretation here than in simulated
                annealing.
        - temp_decay: Multiplicative factor by which temperature is reduced
                on each step. Temperature parameters should be chosen such
                that e^(-cost / temp) never reaches 0.
        - max_neighbors: Number of neighbors generated each round for each
                candidate in the population.
        Returns: best_candidate, best_cost
        The best candidate identified by the search and its cost.

        NOTE: In this case, you should always call random_neighbor(), rather
                than best_neighbor().
        """
        best_state = None
        best_cost = float("inf")
        pop = []
        for i in range(pop_size):
                pop.append(problem.random_candidate())
        temp = init_temp
        for step in range(steps):
                successors = []
                for neighbor in pop:
                        successors.extend(gen_neighbors(problem, neighbor, max_neighbors))
                best_neigh_state, best_neigh_cost = eval_succs(problem, successors)
                if best_neigh_cost < best_cost:
                        best_state = best_neigh_state
                        best_cost = best_neigh_cost
                        print("Best cost: ", best_cost)
                # generate probs for selecting neighbors based on cost
                pop = gen_new_pop(problem, successors, temp, pop_size)
                temp *= temp_decay
        return best_state, best_cost

def gen_neighbors(problem, state, pop_size):
        pop = []
        for i in range(pop_size):
            pop.append(problem.random_neighbor(state))
        return pop    

def eval_succs(problem, succ_list):
        best_state = None
        best_cost = float("inf")
        for succ in succ_list:
                curr_cost = succ[1]
                curr_state = succ[0]
                if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_state = curr_state
        return best_state, best_cost

def gen_new_pop(problem, neighbors, temp, pop_size):
        probs_list = []
        new_pop = []
        normalized = []
        for neighbor in neighbors:
                n_cost = neighbor[1]
                n_prob = exp(-n_cost/temp)
                probs_list.append(n_prob)
        probs_sum = sum(probs_list)
        for prob in probs_list:
                normalized.append(prob/probs_sum)
        # generate new population
        indices = list(range(len(neighbors)))
        choices = choice(indices, pop_size, p=normalized)
        for index in choices:
                new_pop.append(neighbors[index][0])
        return new_pop
        

                
        

