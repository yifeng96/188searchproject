import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
    self.tmpValues = util.Counter();
    iterationsCompleted = 0
    startState = mdp.getStartState();
    while (iterationsCompleted < iterations):
      for state in mdp.getStates():
        self.computeValue(mdp,state,discount)
      for key in self.tmpValues:
        self.values[key] = self.tmpValues[key]
      iterationsCompleted += 1
    
  def computeValue(self,mdp,state,discount):
    actions = mdp.getPossibleActions(state)
    valueList = []
    if (mdp.isTerminal(state)):
      return
    for action in actions:
      transitions = mdp.getTransitionStatesAndProbs(state,action)
      value = 0
      for transition in transitions:
        subValue = float(transition[1]) * (float(mdp.getReward(state,action,transition[0]))+(float(discount)*(float(self.getValue(transition[0])))))
        value += subValue
      valueList.append(value)
    
    self.tmpValues[state] = max(valueList)
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    transitions = self.mdp.getTransitionStatesAndProbs(state,action)
    value = 0
    for transition in transitions:
      subValue = float(transition[1]) * (float(self.mdp.getReward(state,action,transition[0]))+(float(self.discount)*(float(self.getValue(transition[0])))))
      value += subValue
    return value;

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """    
    legalActions = self.mdp.getPossibleActions(state)
    if len(legalActions) == 0: # return None
      return None
    myDict = util.Counter()
    for action in legalActions:
      myDict[action] = self.getQValue(state, action)
    return myDict.argMax()

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
