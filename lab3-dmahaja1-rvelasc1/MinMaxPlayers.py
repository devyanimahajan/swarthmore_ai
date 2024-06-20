########################################
# CS63: Artificial Intelligence, Lab 3
# Fall 2022, Swarthmore College
########################################

class MinMaxPlayer:
    """Gets moves by depth-limited minimax search."""
    def __init__(self, boardEval, depthBound):
        self.name = "MinMax"
        self.boardEval = boardEval   # static evaluation function
        self.depthBound = depthBound # limit of search
        self.bestMove = None         # best move from root

    def getMove(self, game_state):
        """Create a recursive helper function to implement Minimax, and
        call that helper from here. Initialize bestMove to None before
        the call to helper and then return bestMove found."""
        # raise NotImplementedError("TODO")
        # set depth to root
        self.bestMove = None
        self.bounded_min_max(game_state, 0)
        return self.bestMove

    def bounded_min_max(self, state, depth):
        if depth == self.depthBound or state.isTerminal:
            return self.boardEval(state)
        bestValue = state.turn * -float("inf")
        # for each move from state
        for move in state.availableMoves:
            next_state = state.makeMove(move)
            # recursive call
            value = self.bounded_min_max(next_state, depth+1)
            if state.turn > 0:
                if value > bestValue:
                    bestValue = value
                    if depth == 0:
                        self.bestMove = move
            else:
                if value < bestValue:
                    bestValue = value
                    if depth == 0:
                        self.bestMove = move
        return bestValue


class PruningPlayer:
    """Gets moves by depth-limited minimax search with alpha-beta pruning."""
    def __init__(self, boardEval, depthBound):
        self.name = "Pruning"
        self.boardEval = boardEval   # static evaluation function
        self.depthBound = depthBound # limit of search
        self.bestMove = None         # best move from root

    def getMove(self, game_state):
        """Create a recursive helper function to implement AlphaBeta pruning
        and call that helper from here. Initialize bestMove to None before
        the call to helper and then return bestMove found."""
        #raise NotImplementedError("TODO")
        self.bestMove = None
        alpha = -(float("inf"))
        beta = float("inf")
        self.pruning_search(game_state, 0, alpha, beta)
        return self.bestMove

    def pruning_search(self, state, depth, alpha, beta):
        if depth == self.depthBound or state.isTerminal:
            return self.boardEval(state)
        bestValue = state.turn * -float("inf")
        # for each move from state
        for move in state.availableMoves:
            next_state = state.makeMove(move)
            # recursive call
            value = self.pruning_search(next_state, depth+1, alpha, beta)
            if state.turn > 0:
                if value > bestValue:
                    bestValue = value
                    if depth == 0:
                        self.bestMove = move
                alpha = max(value,beta)
            else:
                if value < bestValue:
                    bestValue = value
                    if depth == 0:
                        self.bestMove = move
                beta = min(value, beta)
            if alpha >= beta:
                break
        return bestValue
