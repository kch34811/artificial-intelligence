from util import manhattanDistance
from game import Directions
import random, util

from game import Agent



class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER
    def minimax(agentNumber, depth, gameState):
        # Terminal conditions
        if gameState.isWin() or gameState.isLose() or depth == 0:
          return self.evaluationFunction(gameState), None

        nextAgent = (agentNumber + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth

        # Pac-Man's turn : maximize the score
        if agentNumber == 0:
          value = float('-inf')
          bestAction = None
          for action in gameState.getLegalActions(agentNumber):
            score = minimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action))[0]
            if score > value:
              value, bestAction = score, action
          return value, bestAction
        # Ghosts' turn : minimize the score
        else:
          value = float('inf')
          worstAction = None
          for action in gameState.getLegalActions(agentNumber):
            score = minimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action))[0]
            if score < value:
              value, worstAction = score, action
          return value, worstAction

    # Start minimax from Pacman's turn with full depth
    return minimax(0, self.depth, gameState)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    def minimax(agentNumber, depth, gameState):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)

      nextAgent = (agentNumber + 1) % gameState.getNumAgents()
      nextDepth = depth - 1 if nextAgent == 0 else depth

      actions = gameState.getLegalActions(agentNumber)

      # Maximize for Pac-Man
      if agentNumber == 0:
        return max(minimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, act)) for act in actions)
      # Minimize for ghosts
      else:
        return min(minimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, act)) for act in actions)

    # Start with the first ghost's turn
    return minimax(1, self.depth, gameState.generateSuccessor(0, action))
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def expectimax(agentNumber, depth, gameState):
        # Terminal conditions
        if gameState.isWin() or gameState.isLose() or depth == 0:
          return self.evaluationFunction(gameState), None

        nextAgent = (agentNumber + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth

        # Pac-Man's turn : maximize the score
        if agentNumber == 0:
          value = float('-inf')
          bestAction = None
          for action in gameState.getLegalActions(agentNumber):
            score = expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action))[0]
            if score > value:
              value, bestAction = score, action
          return value, bestAction
        # Ghosts' turn : average the score
        else:
          values = []
          actions = gameState.getLegalActions(agentNumber)
          for action in actions:
            score = expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action))[0]
            values.append(score)
          averageValue = sum(values) / len(values) if values else float('inf')
          return averageValue, None  # Action is None because we don't need to track ghost actions

    # Start expectimax from Pacman's turn with full depth
    return expectimax(0, self.depth, gameState)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def expectimax(agentNumber, depth, gameState):
        if gameState.isWin() or gameState.isLose() or depth == 0:
          return self.evaluationFunction(gameState)

        nextAgent = (agentNumber + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth

        actions = gameState.getLegalActions(agentNumber)
        # Maximize for Pac-Man
        if agentNumber == 0:
          return max(expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, act)) for act in actions)
        # Average for ghosts
        else:
          scores = [expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, act)) for act in actions]
          return sum(scores) / len(scores) if scores else float('inf')

    # Start with the first ghost's turn, after Pac-Man's action
    return expectimax(1, self.depth, gameState.generateSuccessor(0, action))
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def biased_expectimax(agentNumber, depth, gameState):
        # Terminal conditions
        if gameState.isWin() or gameState.isLose() or depth == 0:
          return self.evaluationFunction(gameState), None

        nextAgent = (agentNumber + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth

        # Pac-Man's turn : maximize the score
        if agentNumber == 0:
          value = float('-inf')
          bestAction = None
          for action in gameState.getLegalActions(agentNumber):
            score = biased_expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action))[0]
            if score > value:
              value, bestAction = score, action
          return value, bestAction
        # Ghosts' turn : calculate biased expectation
        else:
          total_value = 0
          actions = gameState.getLegalActions(agentNumber)
          stop_action_prob = 0.5 + 0.5 * 1 / len(actions)
          for action in actions:
            score = biased_expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action))[0]
            if action == Directions.STOP:
              total_value += stop_action_prob * score
            else:
              total_value += (0.5 * 1 / len(actions)) * score
          return total_value, None  # No need to choose an action for the ghosts

    # Start biased_expectimax from Pacman's turn with full depth
    return biased_expectimax(0, self.depth, gameState)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def biased_expectimax(agentNumber, depth, gameState):
        if gameState.isWin() or gameState.isLose() or depth == 0:
          return self.evaluationFunction(gameState)

        nextAgent = (agentNumber + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth

        actions = gameState.getLegalActions(agentNumber)
        # Maximize for Pac-Man
        if agentNumber == 0:
          return max(biased_expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action)) for action in actions)
        # Calculate biased expectation for ghosts
        else:
          totalValue = 0
          probStop = 0.5 + 0.5 / len(actions)  # Probability of stopping
          for action in actions:
            score = biased_expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action))
            if action == Directions.STOP:
              totalValue += score * probStop
            else:
              totalValue += score * (0.5 / len(actions))
          return totalValue  # Return total expected value without normalization

    # Start biased_expectimax calculation from the ghost's turn right after Pac-Man's action
    return biased_expectimax(1, self.depth, gameState.generateSuccessor(0, action))
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def expectiminimax(agentNumber, depth, gameState):
        if gameState.isWin() or gameState.isLose() or depth == 0:
          return self.evaluationFunction(gameState), None

        nextAgent = (agentNumber + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth

        actions = gameState.getLegalActions(agentNumber)

        # Pac-Man's turn : maximize the score
        if agentNumber == 0:
          value = float('-inf')
          bestAction = None
          for action in actions:
            score = expectiminimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action))[0]
            if score > value:
              value, bestAction = score, action
          return value, bestAction
        # Odd-numbered ghost turn : minimize the score
        elif agentNumber % 2 == 1:
          value = float('inf')
          worstAction = None
          for action in actions:
            score = expectiminimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action))[0]
            if score < value:
              value, worstAction = score, action
          return value, worstAction
        # Even-numbered ghost turn : choose uniformly at random
        else:
          totalValue = sum(
            expectiminimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action))[0] for action in actions)
          averageValue = totalValue / len(actions)
          return averageValue, None  # Action doesn't matter for random decision

    # Start expectiminimax from Pacman's turn with full depth
    return expectiminimax(0, self.depth, gameState)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def expectiminimax(agentNumber, depth, gameState):
        if gameState.isWin() or gameState.isLose() or depth == 0:
          return self.evaluationFunction(gameState)

        nextAgent = (agentNumber + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth

        actions = gameState.getLegalActions(agentNumber)
        # Maximize for Pac-Man
        if agentNumber == 0:
          return max(
            expectiminimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, act)) for act in actions)
        # Minimize for odd-numbered ghosts
        elif agentNumber % 2 == 1:
          return min(
            expectiminimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, act)) for act in actions)
        # Average for even-numbered ghosts
        else:
          scores = [expectiminimax(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, act)) for act in actions]
          return sum(scores) / len(scores)

    # Start with the first ghost's turn after Pac-Man's action
    return expectiminimax(1, self.depth, gameState.generateSuccessor(0, action))
    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER
    def alphaBeta(agentNumber, depth, gameState, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None

        nextAgent = (agentNumber + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth

        if agentNumber == 0:  # Pac-Man's turn (maximize)
            value = float('-inf')
            bestAction = None
            for action in gameState.getLegalActions(agentNumber):
                score, _ = alphaBeta(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action), alpha,
                                     beta)
                if score > value:
                    value, bestAction = score, action
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, bestAction
        else:  # Ghosts' turn (minimize for odd, average for even)
            if agentNumber % 2 == 1:  # Odd-numbered ghost, minimize
                value = float('inf')
                for action in gameState.getLegalActions(agentNumber):
                    score, _ = alphaBeta(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action), alpha,
                                         beta)
                    value = min(value, score)
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
                return value, None
            else:  # Even-numbered ghost, expectimax
                totalValue = 0
                actions = gameState.getLegalActions(agentNumber)
                for action in actions:
                    score, _ = alphaBeta(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action), alpha,
                                         beta)
                    totalValue += score
                averageValue = totalValue / len(actions) if actions else float('inf')
                return averageValue, None

    # Start alpha-beta from Pacman's turn with full depth
    return alphaBeta(0, self.depth, gameState, float('-inf'), float('inf'))[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def alphaBeta(agentNumber, depth, gameState, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        nextAgent = (agentNumber + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth

        if agentNumber == 0:  # Pac-Man's turn
            return max(
                alphaBeta(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action), alpha, beta) for
                action in gameState.getLegalActions(agentNumber))
        else:  # Ghosts' turn
            if agentNumber % 2 == 1:  # Minimize for odd-numbered ghosts
                value = float('inf')
                for action in gameState.getLegalActions(agentNumber):
                    score = alphaBeta(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action), alpha,
                                      beta)
                    value = min(value, score)
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
                return value
            else:  # Average for even-numbered ghosts
                scores = [alphaBeta(nextAgent, nextDepth, gameState.generateSuccessor(agentNumber, action), alpha, beta)
                          for action in gameState.getLegalActions(agentNumber)]
                return sum(scores) / len(scores) if scores else float('inf')

    # Start with the first ghost's turn after Pac-Man's action
    return alphaBeta(1, self.depth, gameState.generateSuccessor(0, action), float('-inf'), float('inf'))
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER
  pacmanPosition = currentGameState.getPacmanPosition()
  ghostStates = currentGameState.getGhostStates()
  score = currentGameState.getScore()
  capsules = currentGameState.getCapsules()
  foodList = currentGameState.getFood().asList()

  # Handling food
  if foodList:
      foodDistances = [manhattanDistance(pacmanPosition, food) for food in foodList]
      closestFoodDist = min(foodDistances)
      score += 10 / closestFoodDist  # More weight to the closest food
      score += 4 * sum(1 / f for f in foodDistances)  # Reward proximity to all food

  # Capsules strategy
  if capsules:
      capsuleDistances = [manhattanDistance(pacmanPosition, cap) for cap in capsules]
      closestCapsuleDist = min(capsuleDistances)
      score += 80 / (closestCapsuleDist + 1)  # Higher value for capsules when ghosts are active
  score -= 6 * len(capsules)  # Higher penalty for unused capsules

  # Ghosts proximity and scare management
  ghostPenalty = 0
  for ghost in ghostStates:
      dist = manhattanDistance(pacmanPosition, ghost.getPosition())
      if ghost.scaredTimer > 0:
          score += 200 / dist  # More aggressive chasing when safe to do so
      else:
          if dist <= 1:
              ghostPenalty += 800  # Increased penalty for dangerous proximity
          else:
              ghostPenalty += 12 / dist  # Scaled penalty for proximity

  score -= ghostPenalty

  # Adjust for remaining food and game progression
  score -= 1 * len(foodList)  # Reduced penalty for remaining food to encourage completion

  return score
  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  return 'ExpectimaxAgent'
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction
