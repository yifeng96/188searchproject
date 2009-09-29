from util import manhattanDistance
from game import Directions
import random, util, time

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  
    
  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.
    
    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    
    "Add more of your code here if you want to"
    
    return legalMoves[chosenIndex]
  
  def manhattanDistance( xy1, xy2 ):
    "Returns the Manhattan distance between points xy1 and xy2"
    return abs( xy1[0] - xy2[0] ) + abs( xy1[1] - xy2[1] )
  
  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here. 
    
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.
    
    The code below extracts some useful information from the state, like the 
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    
    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    ghostPositions = currentGameState.getGhostPositions()
    ghostDistances = []
    
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    for pos in ghostPositions:
      ghostDistances.append(manhattanDistance(successorGameState.getPacmanPosition(),pos))
    
    foodPositions = currentGameState.getFood().asList()
    foodDistances = []
    for pos in foodPositions:
      foodDistances.append(manhattanDistance(successorGameState.getPacmanPosition(),pos))
    bestFoodDistance = min(foodDistances)
    bestGhostDistance = min(ghostDistances)
    if (bestGhostDistance != 0):
      #doesn't work very well when your trying to run away from the ghost and are super far away from the food
      if (bestFoodDistance == 0):
        bestFoodDistance = .5
      if (bestGhostDistance >= 2*bestFoodDistance):
        # print "ghost is so far away", bestGhostDistance, " and food is close", bestFoodDistance
        return 1/bestFoodDistance
      
      # print "best food distance is", bestFoodDistance, "food mins is", 1.0/bestFoodDistance, " and max ghost is ", float(-1*bestGhostDistance), "total is " , float(1/bestFoodDistance+1/(-1*bestGhostDistance))
      return float(1/bestFoodDistance)+float(1/(-1*bestGhostDistance))
    else:
      return -1
    
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates() 
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    "*** YOUR CODE HERE ***"
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

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
    
  def maxMove(self,gameState,depth,max_depth,num_ghosts):
    if(max_depth == depth):
      return self.evaluationFunction(gameState)
    else:
      value = -999999999
      bestMove = Directions.STOP
      if (gameState.isLose()):
        return gameState.getScore()
      for action in gameState.getLegalActions(0):
        subValue = self.minMove(gameState.generateSuccessor(0,action),depth,max_depth,num_ghosts,1)
        newValue = max(value,subValue)
        if (newValue > value):
          bestMove = action
        value = newValue
      if (depth != 0):
        return value
      else:
        return bestMove
        
  def minMove(self,gameState,depth,max_depth,num_ghosts,ghostIndex):
    if(max_depth == depth):
      return self.evaluationFunction(gameState)
    else:
      value = 9999999.99
      if (ghostIndex == num_ghosts):
        if (gameState.isLose()):
          value = gameState.getScore()
        for action in gameState.getLegalActions(ghostIndex):
          subValue = self.maxMove(gameState.generateSuccessor(ghostIndex,action),depth+1,max_depth,num_ghosts)
          value = min(value,subValue)
        return value
      else:
        if (gameState.isLose()):
          value = gameState.getScore()
        for action in gameState.getLegalActions(ghostIndex):
          subValue = self.minMove(gameState.generateSuccessor(ghostIndex,action),depth,max_depth,num_ghosts,ghostIndex+1)
          value = min(value,subValue)
        return value

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth 
      and self.evaluationFunction.
      
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
    """
    bestMove = self.maxMove(gameState, 0 , self.depth, gameState.getNumAgents() - 1)
    return bestMove

    
class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  calls = 0

  def maxMove(self,gameState,depth,max_depth,num_ghosts,alpha,beta):
    if(max_depth == depth):
      return self.evaluationFunction(gameState)
    else:
      value = -999999999
      bestMove = Directions.STOP
      if (gameState.isLose()):
        return gameState.getScore()
      for action in gameState.getLegalActions(0):
        subValue = self.minMove(gameState.generateSuccessor(0,action),depth,max_depth,num_ghosts,1,alpha,beta)
        newValue = max(value,subValue)
        if (newValue > value):
          bestMove = action
        value = newValue
        if (value >= beta):
          return value
        alpha = max(alpha,value)
      if (depth != 0):
        return value
      else:
        return bestMove
        
  def minMove(self,gameState,depth,max_depth,num_ghosts,ghostIndex,alpha,beta):
    if(max_depth == depth):
      return self.evaluationFunction(gameState)
    else:
      value = 9999999.99
      if (ghostIndex == num_ghosts):
        if (gameState.isLose()):
          value = gameState.getScore()
        for action in gameState.getLegalActions(ghostIndex):
          subValue = self.maxMove(gameState.generateSuccessor(ghostIndex,action),depth+1,max_depth,num_ghosts,alpha,beta)
          value = min(value,subValue)
          if (value <= alpha):
            return value
          beta = max(beta,value)
        return value
      else:
        if (gameState.isLose()):
          value = gameState.getScore()
        if (gameState.isWin()):
          value = gameState.getScore()
        for action in gameState.getLegalActions(ghostIndex):
          subValue = self.minMove(gameState.generateSuccessor(ghostIndex,action),depth,max_depth,num_ghosts,ghostIndex+1,alpha,beta)
          value = min(value,subValue)
        return value
  
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    bestMove = self.maxMove(gameState, 0 , self.depth, gameState.getNumAgents() - 1,-999999999,99999999)
    return bestMove

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def maxMove(self,gameState,depth,max_depth,num_ghosts,alpha,beta):
    if(max_depth == depth):
      return self.evaluationFunction(gameState)
    else:
      value = -999999999
      bestMove = Directions.STOP
      if (gameState.isLose()):
        return gameState.getScore()
      finalActions = gameState.getLegalActions(0)
      for action in finalActions:
        subValue = self.minMove(gameState.generateSuccessor(0,action),depth,max_depth,num_ghosts,1,alpha,beta)
        newValue = max(value,subValue)

        if (newValue > value):
          bestMove = action
        value = newValue
        if (value >= beta):
          return value
        alpha = max(alpha,value)
      if (depth != 0):
        return value
      else:
        return bestMove
        
  def minMove(self,gameState,depth,max_depth,num_ghosts,ghostIndex,alpha,beta):
    if(max_depth == depth):
      return self.evaluationFunction(gameState)
    else:
      value = 9999999.99
      if (ghostIndex == num_ghosts):
        if (gameState.isLose()):
          value = gameState.getScore()
          return value
        values = []
        for action in gameState.getLegalActions(ghostIndex):
          subValue = self.maxMove(gameState.generateSuccessor(ghostIndex,action),depth+1,max_depth,num_ghosts,alpha,beta)
          values.append(subValue)

        if (float(len(gameState.getLegalActions(ghostIndex))) == 0):
          return 1000
        return float(sum(values))/float(len(gameState.getLegalActions(ghostIndex)))
      else:
        if (gameState.isLose()):
          value = gameState.getScore()
          return value
        values = []
        for action in gameState.getLegalActions(ghostIndex):
          subValue = self.minMove(gameState.generateSuccessor(ghostIndex,action),depth,max_depth,num_ghosts,ghostIndex+1,alpha,beta)
          values.append(subValue)
        if (float(len(gameState.getLegalActions(ghostIndex))) == 0):
          return 1000
        return float(sum(values))/float(len(gameState.getLegalActions(ghostIndex)))
  
  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      
      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    bestMove = self.maxMove(gameState, 0 , self.depth, gameState.getNumAgents() - 1,-9999999999,9999999)
    return bestMove
    

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    
    DESCRIPTION: <write something here so we know what you did>
  """  
  
  foodPositions = currentGameState.getFood().asList()
  foodDistances = []
  currentPos = currentGameState.getPacmanPosition()
  for pos in foodPositions:
    foodDistances.append(manhattanDistance(currentPos,pos))
  bestFoodDistance = min(foodDistances)
  if (bestFoodDistance == 0):
    bestFoodDistance = .5
    
  finalValue = currentGameState.getScore()+float(-bestFoodDistance)+float(-len(currentGameState.getFood().asList()))
  return finalValue
  

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """
    
  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.
      
      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

