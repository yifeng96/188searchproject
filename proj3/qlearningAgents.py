from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
          
class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent
    
    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update
      
    //sample value = Reward of that transition + discount * V 
    //You calculate the sample in the update. 
    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.gamma (discount rate)
    
    Functions you should use
      - self.getLegalActions(state) 
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    self.qValues = util.Counter() # A Counter is a dict with default 0, take q-value tuples
    
    "*** YOUR CODE HERE ***"
  
  def getQValue(self, state, action):
    """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
    "*** YOUR CODE HERE ***"
    return self.qValues[(state,action)]
  
    
  def getValue(self, state):
    """
      Returns max_action Q(state,action)        
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    actions = self.getLegalActions(state)
    qArray = []
    if (len(actions) == 0):
      return 0.0
    
    for action in actions:
      q = self.getQValue(state,action);
      qArray.append(q)
    
    return max(qArray)
    
    
  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    legalActions = self.getLegalActions(state)
    if len(legalActions) == 0: # return None
      return None
    myDict = util.Counter()
    for action in legalActions:
      myDict[action] = self.getQValue(state, action)
    return myDict.argMax()
    
  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.
    
      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """  
    # Pick Action
    legalActions = self.getLegalActions(state)
    if len(legalActions) == 0:
      return None
    # do some extra stuff involving episolen here
    if util.flipCoin(self.epsilon):
      return random.choice(legalActions)
    #else do this
    return self.getPolicy(state)
  
  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a 
      state = action => nextState and reward transition.
      You should do your Q-Value update here
      
      NOTE: You should never call this function,
      it will be called on your behalf
    """
    #possibly do this?
    sample = reward + (self.gamma * self.getValue(nextState))
    newValue = ((1- self.alpha)*self.getQValue(state,action))+(self.alpha*sample)
    self.qValues[(state,action)] = newValue
    
class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"
  
  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
    
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action

    
class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent
     
     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # print self.featExtractor.getFeatures()
    # You might want to initialize weights here.
    self.weightDict = util.Counter()
    
  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    
    return self.featExtractor.getFeatures(state,action)*self.weightDict
    
  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition  
    """
    keys = self.featExtractor.getFeatures(state,action).sortedKeys()
    if (len(self.weightDict) == 0):
      for key in keys:
        self.weightDict[key] = 0
    
    correction = (reward + self.gamma*self.getValue(nextState))-self.getQValue(state,action)
    features = self.featExtractor.getFeatures(state,action)
    for key in keys:
      self.weightDict[key] = self.weightDict[key] + (self.alpha*correction*features[key])
    
    
    
  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)
    
    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass
