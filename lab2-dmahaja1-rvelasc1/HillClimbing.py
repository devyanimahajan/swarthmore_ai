########################################
# CS63: Artificial Intelligence, Lab 2
# Fall 2022, Swarthmore College
########################################

from random import random

def hill_climbing(problem, runs, steps, rand_move_prob):
    """Implementes the hill climbing local search algorithm.
    Inputs:
        - problem: A TSP instance.
        - runs: Number of times to start from a random initial candidate.
        - steps: Number of moves to make in a given run.
        - rand_move_prob: prob of a random neighbor on any given step.
    Returns: best_candidate, best_cost
        The best candidate identified by the search and its cost.

    NOTE: When doing a random move use random_neighbor(), otherwise use
        best_neighbor(). 
    """
    best_state = problem.random_candidate()
    best_cost = problem.cost(best_state)
    for run in range(runs):
        curr_state = problem.random_candidate()
        curr_cost = problem.cost(curr_state)
        for step in range(steps):
            if random() < rand_move_prob:
                curr_state, curr_cost = problem.random_neighbor(curr_state)
            else:
                neighbor_state, neighbor_cost = problem.best_neighbor(curr_state)
                if neighbor_cost < curr_cost:
                    curr_state = neighbor_state
                    curr_cost = neighbor_cost
            if curr_cost < best_cost:
                best_state = curr_state
                best_cost = curr_cost
                print("New Best Cost: ", best_cost)
    return best_state, best_cost
