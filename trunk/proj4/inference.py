import util
import random
import busters
import game

class InferenceModule:
  """
  An inference module tracks a belief distribution over a ghost's location.
  This is an abstract class, which you should not modify.
  """
  
  ############################################
  # Useful methods for all inference modules #
  ############################################
  
  def __init__(self, ghostAgent):
    "Sets the ghost agent for later access"
    self.ghostAgent = ghostAgent
    self.index = ghostAgent.index
    
  def getPositionDistribution(self, gameState):
    """
    Returns a distribution over successor positions of the ghost from the given gameState.
    
    You must first place the ghost in the gameState, using setGhostPosition below.
    """
    ghostPosition = gameState.getGhostPosition(self.index) # The position you set
    actionDist = self.ghostAgent.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
      successorPosition = game.Actions.getSuccessor(ghostPosition, action)
      dist[successorPosition] = prob
    return dist
  
  def setGhostPosition(self, gameState, ghostPosition):
    """
    Sets the position of the ghost for this inference module to the specified
    position in the supplied gameState.
    """
    conf = game.Configuration(ghostPosition, game.Directions.STOP)
    gameState.data.agentStates[self.index] = game.AgentState(conf, False)
    return gameState
  
  def observeState(self, gameState):
    "Collects the relevant noisy distance observation and pass it along."
    distances = gameState.getNoisyGhostDistances()
    if len(distances) >= self.index: # Check for missing observations
      obs = distances[self.index - 1]
      self.observe(obs, gameState)
      
  def initialize(self, gameState):
    "Initializes beliefs to a uniform distribution over all positions."
    # The legal positions do not include the ghost prison cells in the bottom left.
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]   
    self.initializeUniformly(gameState)
    
  ######################################
  # Methods that need to be overridden #
  ######################################
  
  def initializeUniformly(self, gameState):
    "Sets the belief state to a uniform prior belief over all positions."
    pass
  
  def observe(self, observation, gameState):
    "Updates beliefs based on the given distance observation and gameState."
    pass
  
  def elapseTime(self, gameState):
    "Updates beliefs for a time step elapsing from a gameState."
    pass
    
  def getBeliefDistribution(self):
    """
    Returns the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence so far.
    """
    pass

class ExactInference(InferenceModule):
  """
  The exact dynamic inference module should use forward-algorithm
  updates to compute the exact belief function at each time step.
  """
  
  def initializeUniformly(self, gameState):
    "Begin with a uniform distribution over ghost positions."
    self.beliefs = util.Counter()
    for p in self.legalPositions: self.beliefs[p] = 1.0
    self.beliefs.normalize()
  
  def observe(self, observation, gameState):
    """
    Updates beliefs based on the distance observation and Pacman's position.
    
    The noisyDistance is the estimated manhattan distance to the ghost you are tracking.
    
    The emissionModel below stores the probability of the noisyDistance for any true 
    distance you supply.  That is, it stores P(noisyDistance | TrueDistance).
    """
    noisyDistance = observation
    emissionModel = busters.getObservationDistribution(noisyDistance)
    pacmanPosition = gameState.getPacmanPosition()
    
    # Replace this code with a correct observation update
    tmpBeliefs = util.Counter()
    for p in self.legalPositions:
      trueDistance = util.manhattanDistance(p, pacmanPosition)
      tmpBeliefs[p] = self.beliefs[p] * emissionModel[trueDistance]
    tmpBeliefs.normalize()
    self.beliefs = tmpBeliefs;
        
    
  def elapseTime(self, gameState):
    """
    Update self.beliefs in response to a time step passing from the current state.
    
    The transition model is not entirely stationary: it may depend on Pacman's
    current position (e.g., for DirectionalGhost).  
    
    You will need to use two helper methods provided in InferenceModule above:
      1) self.setGhostPosition(gameState, ghostPosition)
          This method alters the gameState by placing the ghost we're tracking
          in a particular position.  This altered gameState can be used to query
          what the ghost would do in this position.
      
      2) self.getPositionDistribution(gameState)
          This method uses the ghost agent to determine what positions the ghost
          will move to from the provided gameState.  The ghost must be placed
          in the gameState with a call to self.setGhostPosition above.
    """

    ghostPositions = util.Counter()
    for p in self.legalPositions:
      self.setGhostPosition(gameState,p)
      for key in self.getPositionDistribution(gameState):
        ghostPositions[p] += self.beliefs[key] * self.getPositionDistribution(gameState)[p]
      
    for key in ghostPositions:
      self.beliefs[key] = ghostPositions[key]
    self.beliefs.normalize()


  def getBeliefDistribution(self):
    return self.beliefs

class ParticleFilter(InferenceModule):
  """
  A particle filter for approximately tracking a single ghost.
  
  Useful helper functions will include random.choice, which chooses
  an element from a list uniformly at random, and util.sample, which
  samples a key from a Counter by treating its values as probabilities.
  """
  
  def initializeUniformly(self, gameState, numParticles=100):
    "Initializes a list of particles."
    self.numParticles = numParticles
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.particles = []
    self.beliefs = util.Counter()
    
    probability = 1.0 / self.numParticles
    for i in range(self.numParticles):
      randomParticle = random.choice(self.legalPositions)
      self.particles.append(randomParticle)
      self.beliefs[randomParticle] += probability
    self.beliefs.normalize()
     

  
  def observe(self, observation, gameState):
    "Update beliefs based on the given distance observation."
    emissionModel = busters.getObservationDistribution(observation)
    pacmanPosition = gameState.getPacmanPosition()
    particle_weights = util.Counter()
    if (gameState.getLivingGhosts()[self.index]):
      keepParticles = False
      for particle in self.particles:
        trueDist = util.manhattanDistance(pacmanPosition, particle)
        weight = emissionModel[trueDist]
        particle_weights[particle] += weight
        keepParticles |= weight > 0
      
      # print "ghost index is ", self.index, "and the paticle weight is " , particle_weights
      if (keepParticles):
        particle_weights.normalize()
        newParticles = []
        for n in range(self.numParticles):
          resample = util.sampleFromCounter(particle_weights)
          newParticles.append(resample)
        self.particles = newParticles
      else:
        self.initializeUniformly(gameState);
    
  
    
  def elapseTime(self, gameState):
    "Update beliefs for a time step elapsing."
    # self.particles = [util.sample(self.getPositionDistribution(self.setGhostPosition(gameState, particle))) for particle in self.particles]
    newParticles = []
    if (gameState.getLivingGhosts()[self.index]):
      for i in range(self.numParticles):
        self.setGhostPosition(gameState, self.particles[i])
        updatedParticle = util.sample(self.getPositionDistribution(gameState))
        newParticles.append(updatedParticle)
      self.particles = newParticles;
    # 
    # sizeOfBoard = (float) (len(self.legalPositions))
    # tmpBeliefs = util.Counter()
    # for position in self.legalPositions:
    #   tmpBeliefs[position] = (float) ((float) (self.particles.count(position)) / sizeOfBoard)
    # tmpBeliefs.normalize();
    # self.beliefs = tmpBeliefs;

  def getBeliefDistribution(self):
    """
    Return the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence and time passage.
    """
    probability = 1.0 / self.numParticles
    beliefs = util.Counter()
    for particle in self.particles:
      beliefs[particle] += probability
    beliefs.normalize()
    return beliefs

class MarginalInference(InferenceModule):
  "A wrapper around the JointInference module that returns marginal beliefs about ghosts."

  def initializeUniformly(self, gameState):
    "Set the belief state to an initial, prior value."
    if self.index == 1: jointInference.initialize(gameState, self.legalPositions)
    jointInference.addGhostAgent(self.ghostAgent)
    
  def observeState(self, gameState):
    "Update beliefs based on the given distance observation and gameState."
    if self.index == 1: jointInference.observeState(gameState)
    
  def elapseTime(self, gameState):
    "Update beliefs for a time step elapsing from a gameState."
    if self.index == 1: jointInference.elapseTime(gameState)
    
  def getBeliefDistribution(self):
    "Returns the marginal belief over a particular ghost by summing out the others."
    jointDistribution = jointInference.getBeliefDistribution()
    dist = util.Counter()
    for t, prob in jointDistribution.items():
      dist[t[self.index - 1]] += prob
    return dist
  
class JointParticleFilter:
  "JointParticleFilter tracks a joint distribution over tuples of all ghost positions."
  
  def initialize(self, gameState, legalPositions, numParticles = 600):
    "Stores information about the game, then initializes particles."
    self.numGhosts = gameState.getNumAgents() - 1
    self.numParticles = numParticles
    self.ghostAgents = []
    self.legalPositions = legalPositions
    self.initializeParticles()
    
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
      newParticle = list(oldParticle) # A list of ghost positions
      for ghostIndex in range(1,len(gameState.getLivingGhosts())):
        if (gameState.getLivingGhosts()[ghostIndex] == True):
          setGhostPositions(gameState, newParticle)
          updatedParticle = util.sample(getPositionDistributionForGhost(gameState, ghostIndex, self.ghostAgents[ghostIndex - 1]))
          newParticle.pop(ghostIndex - 1)
          newParticle.insert(ghostIndex - 1, updatedParticle)
          
      newParticles.append(tuple(newParticle))
    self.particles = newParticles

  
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
    pacmanPosition = gameState.getPacmanPosition()
    noisyDistances = gameState.getNoisyGhostDistances()
    emissionModels = [busters.getObservationDistribution(dist) for dist in noisyDistances]
    
    particleWeightsCounter = util.Counter()
    if (len(emissionModels) != 0):
      for particleArray in self.particles:
          particleArray = list(particleArray)
          particle_weights = util.Counter()
          for ghostIndex in range(self.numGhosts):
            if (noisyDistances[ghostIndex] == 999):
              particleArray[ghostIndex] = (2 * (ghostIndex+1) - 1, 1)
            else:
              trueDist = util.manhattanDistance(pacmanPosition, particleArray[ghostIndex])
              weight = (float) (emissionModels[ghostIndex][trueDist])
              particle_weights[ghostIndex] += weight
          weightProduct = 1
          for particle in particle_weights:
            weightProduct *= particle_weights[particle]
          particleArray = tuple(particleArray)
          particleWeightsCounter[particleArray] = weightProduct
      
      if (sum(particleWeightsCounter.values()) == 0):
        self.initializeParticles()
        return

      newParticles = []
      for n in range(len(self.particles)):
        resample = util.sampleFromCounter(particleWeightsCounter)
        newParticles.append(resample)
      self.particles = newParticles;
    
  
  def getBeliefDistribution(self):
    dist = util.Counter()
    for part in self.particles: dist[part] += 1
    dist.normalize()
    return dist

# One JointInference module is shared globally across instances of MarginalInference 
jointInference = JointParticleFilter()

def getPositionDistributionForGhost(gameState, ghostIndex, agent):
  """
  Returns the distribution over positions for a ghost, using the supplied gameState.
  """
  ghostPosition = gameState.getGhostPosition(ghostIndex) 
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

