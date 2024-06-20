########################################
# CS63: Artificial Intelligence, Lab 4
# Fall 2022, Swarthmore College
########################################

#NOTE: You will probably want to use these imports. Feel free to add more.
from math import log, sqrt
import random

class Node(object):
    """Node used in MCTS"""
    def __init__(self, state):
        self.state = state
        self.children = {} # maps moves to Nodes
        self.visits = 0    # number of times node was in select/expand path
        self.wins = 0      # number of wins for player +1
        self.losses = 0    # number of losses for player +1
        self.value = 0     # value (from player +1's perspective)
        self.untried_moves = list(state.availableMoves) # moves to try

    def updateValue(self, outcome):
        """
        Increments self.visits.
        Updates self.wins or self.losses based on the outcome, and then
        updates self.value.

        This function will be called during the backpropagation phase
        on each node along the path traversed in the selection and
        expansion phases.

        outcome: Who won the game.
                 +1 for a 1st player win
                 -1 for a 2nd player win
                  0 for a draw
        """
        # value = 1 + (wins-losses)/visits
        self.visits += 1
        if outcome > 0:
            self.wins += 1
        elif outcome < 0:
            self.losses += 1
        self.value = 1 + (self.wins - self.losses) / self.visits

    def UCBWeight(self, UCB_const, parent_visits, parent_turn):
        """
        Weight from the UCB formula used by parent to select a child.

        This function calculates the weight for JUST THIS NODE. The
        selection phase, implemented by the MCTSPlayer, is responsible
        for looping through the parent Node's children and calling
        UCBWeight on each.

        UCB_const: the C in the UCB formula.
        parent_visits: the N in the UCB formula.
        parent_turn: Which player is making a decision at the parent node.
           If parent_turn is +1, the stored value is already from the
           right perspective. If parent_turn is -1, value needs to be
           converted to -1's perspective.
        returns the UCB weight calculated
        """
        # v + C x sqrt(ln(N)/n)
        if parent_turn > 0:
            val = self.value
        else:
            val = 2 - self.value
        UCB = val + UCB_const * sqrt(log(parent_visits)/self.visits)
        return UCB

class MCTSPlayer(object):
    """Selects moves using Monte Carlo tree search."""
    def __init__(self, num_rollouts=1000, UCB_const=1.0):
        self.name = "MCTS"
        self.num_rollouts = int(num_rollouts)
        self.UCB_const = UCB_const
        self.nodes = {} # dictionary that maps states to their nodes

    def getMove(self, game_state):
        """Returns best move from the game_state after applying MCTS"""
        #TODO: find existing node in tree or create a node for game_state
        #      and add it to the tree
        #TODO: call MCTS to perform rollouts
        #TODO: return the best move from the current player's perspective
        #raise NotImplementedError("TODO")
        key = str(game_state)
        if key in self.nodes:
            curr_node = self.nodes[key]
        else:
            self.nodes[key] = Node(game_state)
            curr_node = self.nodes[key]
        self.MCTS(curr_node)
        bestValue = -float("inf")
        bestMove = None
        for move, child_node in curr_node.children.items():
            print("testing move")
            print(move)
            if child_node.state.turn == 1:
                value = child_node.value
            else:
                value = 2 - child_node.value
            if value > bestValue:
                bestValue = value
                bestMove = move
        return bestMove

    def status(self, node):
        """
        This method is used solely for debugging purposes. Given a
        node in the MCTS tree, reports on the node's data (wins, losses,
        visits, values), as well as the data of all of its immediate
        children. Helps to verify that MCTS is working properly.
        Returns: None
        """
        #raise NotImplementedError("TODO")
        print("node wins %d, losses %d, visits %d, value %d" % \
        (node.wins, node.losses, node.visits, node.value))

    def MCTS(self, current_node):
        """
        Plays out random games from the current node to a terminal state.
        Each rollout consists of four phases:
        1. Selection: Nodes are selected based on the max UCB weight.
                      Ends when a node is reached where not all children
                      have been expanded.
        2. Expansion: A new node is created for a random unexpanded child.
        3. Simulation: Uniform random moves are played until end of game.
        4. Backpropagation: Values and visits are updated for each node
                     on the path traversed during selection and expansion.
        Returns: None
        """
        #TODO: Create helper functions for each phase
        #TODO: selection
        #TODO: expansion
        #TODO: simulation
        #TODO: backpropagation
        #TODO: after all rollouts completed, call status on current_node
        #      to view a summary of results
        # raise NotImplementedError("TODO")
        for roll in range(self.num_rollouts):
            path = self.selection(current_node)
            selected_node = path[-1]
            if selected_node.state.isTerminal:
                outcome = selected_node.state.winner
            else:
                next_node = self.expansion(selected_node)
                path.append(next_node)
                outcome = self.simulation(next_node.state)
            self.backpropagation(path, outcome)
        self.status(current_node)

    def selection(self, curr_node, path=[]):
        if len(curr_node.untried_moves) > 0:
            # not fully expanded, at the frontier
            path.append(curr_node)
            return path
        else:
            # fully expanded
            best_UCB = 0
            best_child = None
            for move in curr_node.untried_moves:
                next_UCB = curr_node.children[move].UCBWeight(self.UCB_const, \
                    curr_node.visits, curr_node.state.turn)
                if next_UCB > best_UCB:
                    best_UCB = next_UCB
                    best_child = curr_node.children[move]
            path.append(best_child)
            print(type(best_child))
            return self.selection(best_child, path)

    def expansion(self, curr_node):
        if not curr_node.state.isTerminal:
            move = curr_node.untried_moves[0]
            child = Node(curr_node.state.makeMove(move))
            key = str(child.state)
            self.nodes[key] = child
            return child
        else:
            return curr_node


    def simulation(self, state):
        while not state.isTerminal:
            movelst = len(state.availableMoves)
            ind = random.randrange(movelst)
            move = state.availableMoves[ind]
            new_state = state.makeMove(move)
            state = new_state
        return state.winner



    def backpropagation(self, path, outcome):
        #raise NotImplementedError("TODO")
        for node in reversed(path):
            node.updateValue(outcome)
            newval = node.value
            outcome = newval
