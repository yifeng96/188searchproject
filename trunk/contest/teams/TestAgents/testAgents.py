from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Directions
import keyboardAgents
import game
from util import nearestPoint
from util import Counter
import capture

SONAR_MAX = (capture.SONAR_NOISE_RANGE - 1)/2
SONAR_DENOMINATOR = 2 ** SONAR_MAX  + 2 ** (SONAR_MAX + 1) - 2.0
SONAR_NOISE_PROBS = [2 ** (SONAR_MAX-abs(v)) / SONAR_DENOMINATOR  for v in capture.SONAR_NOISE_VALUES]


#############
# FACTORIES #
#############

NUM_KEYBOARD_AGENTS = 0
class TestAgents(AgentFactory):
  
  def __init__(self, isRed, first='offense', second='defense', rest='offense'):
    AgentFactory.__init__(self, isRed)
    self.agents = [first, second]
    self.rest = rest

  def getAgent(self, index):
    if len(self.agents) > 0:
      return self.choose(self.agents.pop(0), index)
    else:
      return self.choose(self.rest, index)

  def choose(self, agentStr, index):
    if agentStr == 'offense':
      return OffensiveReflexAgent_us(index)
    if agentStr == 'defense':
      return DefensiveReflexAgent_us(index)
  
  def getAgent(self, index):
    if len(self.agents) > 0:
      return self.choose(self.agents.pop(0), index)
    else:
      return self.choose(self.rest, index)

class randAgent(CaptureAgent):
    def chooseAction(self, gameState):
        print "AGENT #: ", self.index
        print "Actions: ", gameState.getLegalActions(self.index)
        print "Noisy distances: ", gameState.getAgentDistances()
        #print gameState.getAgentPosition(0)
        #print gameState.getAgentPosition(1)
        #print gameState.getAgentPosition(2)
        #print gameState.getAgentPosition(3)
        agent0Pos = gameState.getAgentPosition(0)
        agent2Pos = gameState.getAgentPosition(2)
        print "positions: ", agent0Pos, agent2Pos
        trueDistToPartner = self.getMazeDistance(agent0Pos, agent2Pos)
        print "True dist to partner: ", trueDistToPartner
        for i in range(-7, 7):
          print gameState.getDistanceProb(gameState.getAgentDistances()[1] + i, gameState.getAgentDistances()[1])
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)
        
class ReflexCaptureAgent_us(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def registerInitialState(self, gameState):
    self.red = gameState.isOnRedTeam(self.index)
    self.distancer = distanceCalculator.Distancer(gameState.data.layout)
    self.enemyTeam = self.getEnemyTeam(gameState, self.red)
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    jointInference.initialize(gameState, self.legalPositions, self.red, 300)
    import __main__
    if '_display' in dir(__main__):
      self.display = __main__._display
    self.enemyBeliefs = util.Counter()
    
    self.firstMove = True
  
  def getEnemyTeam(self, gameState, isRed):
    if isRed == True:
      return gameState.getBlueTeamIndices()
    else:
      return gameState.getRedTeamIndices()
  
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    #start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    #print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    
    maxValue = max(values)
    self.updateBeliefs(gameState)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    print self.updateBeliefs(gameState)
    return random.choice(bestActions)
  
  def updateBeliefs(self, gameState):
    if not self.firstMove: jointInference.elapseTime(gameState)
    self.firstMove = False
    jointInference.observeState(gameState)
    self.enemyBeliefs = jointInference.getBeliefDistribution()
  
  

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent_us(ReflexCaptureAgent_us):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent_us(ReflexCaptureAgent_us):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0
    print "Action: ", action
    print "Current: ", gameState.getAgentDistances(), ". Successor: ", successor.getAgentDistances()
    print "get Opponents: ", [successor.getAgentState(i).getPosition() for i in self.getOpponents(successor)]
    print self.getOpponents(successor)[0]
    #for i in range(-10, 10):
    #  print "", i, ": ", gameState.getDistanceProb(i, self.getOpponents(successor)[0])
    
    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


class JointParticleFilter:
  "JointParticleFilter tracks a joint distribution over tuples of all ghost positions."
  
  def initialize(self, gameState, legalPositions, isRed, numParticles = 600):
    "Stores information about the game, then initializes particles."
    self.numGhosts = gameState.getNumAgents() - 1
    self.numParticles = numParticles
    self.ghostAgents = []
    self.legalPositions = legalPositions
    self.initializeParticles()
    self.beliefs = util.Counter()
    self.isRed = isRed
    if isRed == True:
      self.enemyTeam = [x for x in gameState.getBlueTeamIndices()]
      self.ourTeam = [x for x in gameState.getRedTeamIndices()]
    else:
      self.enemyTeam = [x for x in gameState.getRedTeamIndices()]
      self.ourTeam = [x for x in gameState.getBlueTeamIndices()]
    
  def initializeParticles(self):
    "Initializes particles randomly.  Each particle is a tuple of ghost positions."
    self.particles = []
    for i in range(self.numParticles):
      self.particles.append(tuple([random.choice(self.legalPositions) for j in range(self.numGhosts)]))

  def addGhostAgent(self, agent):
    "Each ghost agent is registered separately and stored (in case they are different)."
    self.ghostAgents.append(agent)
    
  def elapseTime(self, gameState):
    """
    Samples each particle's next state based on its current state and the gameState.
    
    You will need to use two helper methods provided below:
      1) setGhostPositions(gameState, ghostPositions)
          This method alters the gameState by placing the ghosts in the supplied positions.
      
      2) getPositionDistributionForGhost(gameState, ghostIndex, agent)
          This method uses the supplied ghost agent to determine what positions 
          a ghost (ghostIndex) controlled by a particular agent (ghostAgent) 
          will move to in the supplied gameState.  All ghosts
          must first be placed in the gameState using setGhostPositions above.
          Remember: ghosts start at index 1 (Pacman is agent 0).  
          
          The ghost agent you are meant to supply is self.ghostAgents[ghostIndex-1],
          but in this project all ghost agents are always the same.
    """
    newParticles = []
    for oldParticle in self.particles:
      newParticle = util.Counter()
      for enemyIndex in range(len(self.enemyTeam)):
        setGhostPositions(gameState, oldParticle)
        updatedParticle = util.sample(getPositionDistributionForGhost(gameState, enemyIndex, gameState.getAgentState(enemyIndex)))
        newParticle[enemyIndex] = updatedParticle
      newParticles.append(tuple(newParticle.values()))
    self.particles = newParticles
    
    dist = util.Counter()
    for part in self.particles: dist[part] += 1
    dist.normalize()
    self.beliefs = dist
    
  def observeState(self, gameState):
    """
    Resamples the set of particles using the likelihood of the noisy observations.
    
    A correct implementation will handle two special cases:
      1) When a ghost is captured by Pacman, all particles should be updated so
          that the ghost appears in its cell, position (2 * ghostIndex - 1, 1).
          Captured ghosts always have a noisyDistance of 999.
         
      2) When all particles receive 0 weight, they should be recreated from the
          prior distribution by calling initializeParticles.  
    """ 
    agentPosition = gameState.getAgentPosition(self.ourTeam[0])
    noisyDistances = []
    for i in self.enemyTeam:
      noisyDistances.append(gameState.getAgentDistances()[i])
    
    if len(noisyDistances) < len(self.enemyTeam): return
    emissionModels = [getObservationDistribution(dist) for dist in noisyDistances]
    particleWeightsCounter = util.Counter()
    
    if (len(emissionModels) != 0):
      
      for particle in self.particles:
        updatedParticle = []
        ghost_weights = util.Counter()
        for enemyIndex in range(len(self.enemyTeam)):
          updatedParticle.insert(enemyIndex, particle[enemyIndex])
          trueDist = util.manhattanDistance(agentPosition, particle[enemyIndex])
          ghost_weights[enemyIndex] = (float) (emissionModels[enemyIndex][trueDist])
        weightProduct = 1
        for i in range(len(self.enemyTeam)):
          weightProduct = weightProduct * ghost_weights[i]
        particleWeightsCounter[tuple(updatedParticle)] = weightProduct
      
      if (sum(particleWeightsCounter.values()) == 0):
        self.initializeParticles()
        return
      newParticles = []
      for n in range(self.numParticles):
        resample = util.sample(particleWeightsCounter)
        newParticles.append(resample)
      self.particles = newParticles;
      
      dist = util.Counter()
      for part in self.particles: dist[part] += 1
      dist.normalize()
      self.beliefs = dist


  def getBeliefDistribution(self):
    dist = util.Counter()
    for part in self.particles: dist[part] += 1
    dist.normalize()
    return dist
    
  
  
def getObservationDistribution(noisyDistance):
  """
  Returns the factor P( noisyDistance | TrueDistances ), the likelihood of the provided noisyDistance
  conditioned upon all the possible true distances that could have generated it.
  """
  observationDistributions = {}
  if noisyDistance not in observationDistributions:
    distribution = util.Counter()
    for error , prob in zip(capture.SONAR_NOISE_VALUES, SONAR_NOISE_PROBS):
      distribution[max(1, noisyDistance - error)] += prob
    observationDistributions[noisyDistance] = distribution
  return observationDistributions[noisyDistance]
    
# One JointInference module is shared globally across instances of MarginalInference 
jointInference = JointParticleFilter()

def getPositionDistributionForGhost(gameState, ghostIndex, agent):
  """
  Returns the distribution over positions for a ghost, using the supplied gameState.
  """
  ghostPosition = gameState.getAgentPosition(ghostIndex) 
  actionDist = agent.getDistribution(gameState)
  dist = util.Counter()
  for action, prob in actionDist.items():
    successorPosition = game.Actions.getSuccessor(ghostPosition, action)
    dist[successorPosition] = prob
  return dist
  
def setGhostPositions(gameState, ghostPositions):
  "Sets the position of all ghosts to the values in ghostPositionTuple."
  for index, pos in enumerate(ghostPositions):
    conf = game.Configuration(pos, game.Directions.STOP)
    gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
  return gameState  















###
# Baseline agents
###

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0
    print "Action: ", action
    print "Current: ", gameState.getAgentDistances(), ". Successor: ", successor.getAgentDistances()
    print "get Opponents: ", [successor.getAgentState(i).getPosition() for i in self.getOpponents(successor)]
    print self.getOpponents(successor)[0]
    #for i in range(-10, 10):
    #  print "", i, ": ", gameState.getDistanceProb(i, self.getOpponents(successor)[0])
    
    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}









class expectimaxAgent(CaptureAgent):
  def __init__( self, index, timeForComputing = .1 ):
    """
    Lists several variables you can query:
    self.index = index for this agent
    self.red = true if you're on the red team, false otherwise
    self.agentsOnTeam = a list of agent objects that make up your team
    self.distancer = distance calculator (contest code provides this)
    self.observationHistory = list of GameState objects that correspond
        to the sequential order of states that have occurred so far this game
    self.timeForComputing = an amount of time to give each turn for computing maze distances
        (part of the provided distance calculator)
    """
    # Agent index for querying state
    self.index = index

    # Whether or not you're on the red team
    self.red = None

    # Agent objects controlling you and your teammates
    self.agentsOnTeam = None

    # Maze distance calculator
    self.distancer = None

    # A history of observations
    self.observationHistory = []

    # Time to spend each turn on computing maze distances
    self.timeForComputing = timeForComputing

    # Access to the graphics
    self.display = None
    
    self.counter = Counter()
    
  def chooseAction(self, gameState):
    result = self.counter.argMax(self.expectimax(gameState, 2, self.index))
    self.counter.clear()
    return result
  
  def expectimax(self, state, depth, agent):
    print "agent num: ", agent
    if depth == 0:
      return self.evaluationFunction(state)
    if agent == self.index:             # Current agent's turn
      legalMoves = state.getLegalActions(agent)
      print "legalMoves: ", legalMoves
      if (len(legalMoves) == 0) or (state.isOver()): # Winning states
        return self.evaluationFunction(state)
      children = []
      for move in legalMoves:
        children.append((state.generateSuccessor(self.index, move), move))
      a = -1e308
      for child in children:
        a = max(a, self.expectimax(child[0], depth, 2))
        self.counter[move] = a
      return a
    else:
      if agent in self.getTeam(state):        # A teammate's turn
        legalMoves = state.getLegalActions(agent)
        if (len(legalMoves) == 0) or (state.isOver()): # Winning states
          return self.evaluationFunction(state)
        children = []
        for move in legalMoves:
          children.append((state.generateSuccessor(agent, move), move))
        a = -1e308
        for child in children:
          a = max(a, self.expectimax(child[0], depth, agent+1))
          #self.counter[move] = self.counter[move] + a
        return a
      else:                                   # An enemies' turn
        if agent == (state.getNumAgents()):
          return self.expectimax(state, depth - 1, 0)
        legalMoves = state.getLegalActions(agent)
        if (len(legalMoves) == 0) or (state.isOver()):
          return self.evaluationFunction(state)
        children = []
        a = 1e308
        for move in legalMoves:
          children.append((state.generateSuccessor(agent, move), move))
        for child in children:
          if agent == (state.getNumAgents()):
            a = min(self.expectimax(child[0], depth+1, 0))
            #self.counter[move] = self.counter[move] + 
          else:
            a = max(self.expectimax(child[0], depth, agent+2))  
        return a / (len(legalMoves))
  
  def getMax(self, tuples): # tuple = [ value, action ]
    bestVal = max([tuple[0] for tuple in tuples])
    bestTuples = [tuple for tuple in tuples if tuple[0] == bestVal]
    return bestTuples[0]