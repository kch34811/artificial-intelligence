from typing import Callable, List, Set

import shell
import util
import wordsegUtil


############################################################
# Problem 1a: Solve the segmentation problem under a unigram model

class WordSegmentationProblem(util.SearchProblem):
    def __init__(self, query: str, unigramCost: Callable[[str], float]):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == len(self.query)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return [(self.query[state:end], end, self.unigramCost(self.query[state:end]))
                for end in range(state + 1, len(self.query) + 1)]
        # END_YOUR_CODE


def segmentWords(query: str, unigramCost: Callable[[str], float]) -> str:
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(WordSegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ' '.join(ucs.actions) if ucs.actions else ''
    # END_YOUR_CODE


############################################################
# Problem 1b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords: List[str], bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.queryWords)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        index, prevWord = state
        currentWordOptions = self.possibleFills(self.queryWords[index]) or {self.queryWords[index]}
        return [(word, (index + 1, word), self.bigramCost(prevWord, word))
                for word in currentWordOptions]
    # END_YOUR_CODE


def insertVowels(queryWords: List[str], bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    return ' '.join(ucs.actions) if ucs.actions else ''
    # END_YOUR_CODE


############################################################
# Problem 1c: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query: str, bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.query)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        index, prevWord = state
        return [(fill, (end, fill), self.bigramCost(prevWord, fill))
                for end in range(index + 1, len(self.query) + 1)
                for fill in (self.possibleFills(self.query[index:end]) or {self.query[index:end]})]
        # END_YOUR_CODE


def segmentAndInsert(query: str, bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    return ' '.join(ucs.actions) if ucs.actions else ''
    # END_YOUR_CODE


############################################################
# Problem 2a: Solve the maze search problem with uniform cost search

class MazeProblem(util.SearchProblem):
    def __init__(self, start: tuple, goal: tuple, moveCost: Callable[[tuple, str], float],
            possibleMoves: Callable[[tuple], Set[tuple]]) -> float:
        self.start = start
        self.goal = goal
        self.moveCost = moveCost
        self.possibleMoves = possibleMoves

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.start
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == self.goal
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return [(action, nextState, self.moveCost(state, action))
                for action, nextState in self.possibleMoves(state)]
        # END_YOUR_CODE
            

def UCSMazeSearch(start: tuple, goal: tuple, moveCost: Callable[[tuple, str], float],
            possibleMoves: Callable[[tuple], Set[tuple]]) -> float:
    ucs = util.UniformCostSearch()
    ucs.solve(MazeProblem(start, goal, moveCost, possibleMoves))
    
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    if ucs.actions is None:
        return float('inf')
    path_cost = ucs.totalCost
    return path_cost
    # END_YOUR_CODE


############################################################
# Problem 2b: Solve the maze search problem with A* search

def consistentHeuristic(goal: tuple):
    def _consistentHeuristic(state: tuple) -> float:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return abs(state[0] - goal[0]) + abs(state[1] - goal[1])
        # END_YOUR_CODE
    return _consistentHeuristic

def AStarMazeSearch(start: tuple, goal: tuple, moveCost: Callable[[tuple, str], float],
            possibleMoves: Callable[[tuple], Set[tuple]]) -> float:
    ucs = util.UniformCostSearch()
    ucs.solve(MazeProblem(start, goal, moveCost, possibleMoves), heuristic=consistentHeuristic(goal))
    
    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    if ucs.actions is None:
        return float('inf')  # Return an infinite cost if no path is found

    pathCost = 0
    currentState = start

    for action in ucs.actions:
        # Get the next state from the current state and action
        nextState = [newState for move, newState in possibleMoves(currentState) if move == action][0]

        # Compute the cost from current state to next state
        action_cost = moveCost(currentState, action)

        # Update the total path cost
        pathCost += action_cost

        # Update the current state to the next state
        currentState = nextState

    # Check if the current state is at the goal
    if currentState != goal:
        pathCost += moveCost(currentState, goal)  # Add the cost to reach the goal from the last state

    return pathCost
    # END_YOUR_CODE

############################################################


if __name__ == '__main__':
    shell.main()
