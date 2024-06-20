########################################
# CS63: Artificial Intelligence, Lab 2
# Fall 2022, Swarthmore College
########################################

from math import exp
from random import random

def simulated_annealing(problem, runs, steps, init_temp, temp_decay):
    """Implementes the simulated annealing local search algorithm.
    Inputs:
        - problem: A TSP instance.
        - runs: Number of times to start from a random initial candidate.
        - steps: Number of moves to make in a given run.
        - init_temp: Initial temperature for the start of each run.
                This should scale linearly relative to the cost of a
                typical candidate.
        - temp_decay: Multiplicative factor by which temperature is reduced
                on each step.
    Returns: best_candidate, best_cost
        The best candidate identified by the search and its cost.

    NOTE: In this case, you should always call random_neighbor(), rather
          than best_neighbor().
    """
    #raise NotImplementedError("TODO")
    best_state = problem.random_candidate()
    best_cost = problem.cost(best_state)
    for run in range(runs):
        temp = init_temp
        curr_state = problem.random_candidate()
        curr_cost = problem.cost(curr_state)
        for step in range(steps):
            neighbor_state, neighbor_cost = problem.random_neighbor(curr_state)
            delta = curr_cost - neighbor_cost
            if delta >0 or random() < exp(delta/temp):
                curr_state = neighbor_state
                curr_cost = neighbor_cost
            if curr_cost < best_cost:
                best_state = curr_state
                best_cost = curr_cost
                # print("New best state:", best_state)
                print("New best cost:", best_cost)
            temp *= temp_decay
    return best_state, best_cost
