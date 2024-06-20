########################################
# CS63: Artificial Intelligence, Lab 3
# Fall 2022, Swarthmore College
########################################

import numpy as np
from random import choice
from Mancala import Mancala
from Breakthrough import Breakthrough

def mancalaBasicEval(mancala_game):
    """Difference between the scores for each player.
    Returns +(max possible score) if player +1 has won.
    Returns -(max possible score) if player -1 has won.

    Otherwise returns (player +1's score) - (player -1's score).

    Remember that the number of houses and seeds may vary."""
    if mancala_game.isTerminal:
        winner = mancala_game.winner
        # MAX wins
        if winner > 0:
            return 48
        # MIN wins
        elif winner < 0:
            return -48
        # Tie
        else:
            return 0
    return mancala_game.scores[0] - mancala_game.scores[1]

def breakthroughBasicEval(breakthrough_game):
    """Measures how far each player's pieces have advanced
    and returns the difference.

    Returns +(max possible advancement) if player +1 has won.
    Returns -(max possible advancement) if player -1 has won.

    Otherwise finds the rank of each piece (number of rows onto the board it
    has advanced), sums these ranks for each player, and
    returns (player +1's sum of ranks) - (player -1's sum of ranks).

    An example on a 5x3 board:
    ------------
    |  0  1  1 |  <-- player +1 has two pieces on rank 1
    |  1 -1  1 |  <-- +1 has two pieces on rank 2; -1 has one piece on rank 4
    |  0  1 -1 |  <-- +1 has (1 piece * rank 3); -1 has (1 piece * rank 3)
    | -1  0  0 |  <-- -1 has (1*2)
    | -1 -1 -1 |  <-- -1 has (3*1)
    ------------
    sum of +1's piece ranks = 1 + 1 + 2 + 2 + 3 = 9
    sum of -1's piece ranks = 1 + 1 + 1 + 2 + 3 + 4 = 12
    state value = 9 - 12 = -3

    Remember that the height and width of the board may vary."""
    #raise NotImplementedError("TODO")
    height = len(breakthrough_game.board)
    width = len(breakthrough_game.board[0])

    if breakthrough_game.isTerminal:
        winner = breakthrough_game.winner
        if winner<0:
            return -9999999999
        elif winner>0:
            return 99999999999
        else:
            return 0
    else:
        pos = 0
        neg = 0
        for i in range(height):
            for j in range(width):
                current_inc = i + 1
                if breakthrough_game.board[i,j] < 0:
                    val = height - i
                    neg += val
                elif breakthrough_game.board[i,j] > 0:
                    pos += current_inc
        return pos-neg


def breakthroughBetterEval(breakthrough_game):
    """A heuristic that generally wins agains breakthroughBasicEval.
    This must be a static evaluation function (no search allowed).

    TODO: Update this comment to describe your heuristic."""
    height = len(breakthrough_game.board)
    width = len(breakthrough_game.board[0])
    #determine whether or not the game has been won, then determine winner
    if breakthrough_game.isTerminal:
        winner = breakthrough_game.winner
        if winner<0:
            return -9999999999
        elif winner>0:
            return 99999999999
        else:
            return 0
    #evaluate position weight
    else:
        """
        pos_halfway = 0
        neg_halfway = 0
        half_height = height//2
        """
        pos = 0
        neg = 0
        """
        pos_pieces = 0
        neg_pieces = 0
        """
        for i in range(height):
            for j in range(width):
                current_inc = i + 1
                if breakthrough_game.board[i,j] < 0:
                    val = height - i
                    neg += val
                    if i+1 < height and j+1 < width and j-1 >= 0:
                        if breakthrough_game.board[i+1,j+1] > 0 or breakthrough_game.board[i+1,j-1] > 0:
                            neg += 5
                        elif breakthrough_game.board[i+1,j] > 0:
                            neg -= 5
                    """
                    neg_pieces += 1
                    if i <= half_height:
                        neg_halfway += val
                    """
                elif breakthrough_game.board[i,j] > 0:
                    pos += current_inc
                    if i-1 >= 0 and j+1 < width and j-1 >= 0:
                        if breakthrough_game.board[i-1,j+1] < 0 or breakthrough_game.board[i-1,j-1] < 0:
                            pos *= 2
                        elif breakthrough_game.board[i-1,j] < 0:
                            pos -= 5
                    """
                    pos_pieces += 1
                    if i >= half_height:
                        pos_halfway += current_inc
                else:
                    if i>=half_height:
                        pos_halfway += 1
                    else:
                        neg_halfway += 1
        half_score = pos_halfway - neg_halfway
        """
        score = pos - neg
        return score


if __name__ == '__main__':
    """
    Create a game of Mancala.  Try 10 random moves and check that the
    heuristic is working properly.
    """
    """
    print("\nTESTING MANCALA HEURISTIC")
    print("-"*50)
    game1 = Mancala()
    print(game1)
    for i in range(10):
        move = choice(game1.availableMoves)
        print("\nmaking move", move)
        game1 = game1.makeMove(move)
        print(game1)
        score = mancalaBasicEval(game1)
        print("basicEval score", score)
    """

    # Add more testing for the Breakthrough
    print("\nTESTING BREAKTHROUGH HEURISTIC")
    print("-"*50)
    game2 = Breakthrough(5,3)
    print(game2)
    for i in range(10):
        move = choice(game2.availableMoves)
        print("\nmaking move", move)
        game2 = game2.makeMove(move)
        print(game2)
        score = breakthroughBasicEval(game2)
        print("basicEval score", score)
        score2 = breakthroughBetterEval(game2)
        print("betterEval score", score2)
