#! /usr/bin/env python3
########################################
# CS63: Artificial Intelligence, Lab 1
# Fall 2022, Swarthmore College
########################################

import TrafficJam
import FifteenPuzzle

def testBlockingHeuristic():
    puzzle = TrafficJam.read_puzzle("puzzles/traffic01.txt")
    result = TrafficJam.blockingHeuristic(puzzle)
    assert result == 4, "traffic01: expected 4 but got " + str(result)

    puzzle = TrafficJam.read_puzzle("puzzles/traffic02.txt")
    result = TrafficJam.blockingHeuristic(puzzle)
    assert result == 6, "traffic02: expected 6 but got " + str(result)

    puzzle = TrafficJam.read_puzzle("puzzles/traffic03.txt")
    result = TrafficJam.blockingHeuristic(puzzle)
    assert result == 4, "traffic03: expected 4 but got " + str(result)

    puzzle = TrafficJam.read_puzzle("puzzles/traffic04.txt")
    result = TrafficJam.blockingHeuristic(puzzle)
    assert result == 4, "traffic04: expected 4 but got " + str(result)

    print("All blocking tests passed!")

def testBetterHeuristic():

    #print("TODO: implement better heuristic tests")
    puzzle = TrafficJam.read_puzzle("puzzles/traffic01.txt")
    result = TrafficJam.betterHeuristic(puzzle)
    assert result == 5, "traffic01: expected 5 but got " + str(result)

    puzzle = TrafficJam.read_puzzle("puzzles/traffic02.txt")
    result = TrafficJam.betterHeuristic(puzzle)
    assert result == 8, "traffic02: expected 8 but got " + str(result)

    puzzle = TrafficJam.read_puzzle("puzzles/traffic03.txt")
    result = TrafficJam.betterHeuristic(puzzle)
    assert result == 7, "traffic03: expected 7 but got " + str(result)

    print("All better tests passed!")


def testDisplacedHeuristic():
    puzzle = FifteenPuzzle.read_puzzle("puzzles/fifteen00.json")
    result = FifteenPuzzle.displacedHeuristic(puzzle)
    assert result == 3, "fifteen00: expected 3 but got " + str(result)

    puzzle = FifteenPuzzle.read_puzzle("puzzles/fifteen01.json")
    result = FifteenPuzzle.displacedHeuristic(puzzle)
    assert result == 8, "fifteen01: expected 8 but got " + str(result)

    puzzle = FifteenPuzzle.read_puzzle("puzzles/fifteen02.json")
    result = FifteenPuzzle.displacedHeuristic(puzzle)
    assert result == 8, "fifteen02: expected 8 but got " + str(result)

    puzzle = FifteenPuzzle.read_puzzle("puzzles/fifteen03.json")
    result = FifteenPuzzle.displacedHeuristic(puzzle)
    assert result == 14, "fifteen03: expected 14 but got " + str(result)

    print("All displaced tests passed!")


def testManhattanHeuristic():
    puzzle = FifteenPuzzle.read_puzzle("puzzles/fifteen00.json")
    result = FifteenPuzzle.manhattanHeuristic(puzzle)
    assert result == 4, "fifteen00: expected 4 but got " + str(result)

    puzzle = FifteenPuzzle.read_puzzle("puzzles/fifteen01.json")
    result = FifteenPuzzle.manhattanHeuristic(puzzle)
    assert result == 16, "fifteen01: expected 16 but got " + str(result)

    print("TODO: Implement additional manhattan heuristic tests")

    print("All manhattan tests passed!")

if __name__ == '__main__':
    testBlockingHeuristic()
    testBetterHeuristic()
    testDisplacedHeuristic()
    testManhattanHeuristic()
