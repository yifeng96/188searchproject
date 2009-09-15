"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 74].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.18].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
  "*** YOUR CODE HERE ***"  
  seenAlready = {}
  startState = problem.getStartState();
  state = [startState,"",1,[]];
  theStack = util.Stack();
  seenAlready[state[0]] = "1"
  path = []

  if (problem.isGoalState(startState)) {
    return []
  }
  
  while(1):    
    children = problem.getSuccessors(state[0]);
    
    if (children):
      for item in children:
        if (item[0] not in seenAlready):
          tmpPath = list(state[3]);
          tmpPath.append(item[1])
          item = list(item);
          item.append(tmpPath)
          theStack.push(item)
          seenAlready[item[0]] = "1"
        if (problem.isGoalState(item[0])):
          return item[3]
    
    state = theStack.pop();

  return []

def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first. [p 74]"
  "*** YOUR CODE HERE ***"
  seenAlready = {}
  startState = problem.getStartState();
  state = [startState,"",1,[]];
  theQueue = util.Queue();
  seenAlready[state[0]] = "1"
  path = []

  if (problem.isGoalState(startState)) {
    return []
  }
  
  while(1):    
    children = problem.getSuccessors(state[0]);
    
    if (children):
      for item in children:
        if (item[0] not in seenAlready):
          print "not seen"
          tmpPath = list(state[3]);
          tmpPath.append(item[1])
          item = list(item);
          item.append(tmpPath)
          theQueue.push(item)
          seenAlready[item[0]] = "1"
        if (problem.isGoalState(item[0])):
          if (len(item) < 4):
            tmpPath = list(state[3])
            tmpPath.append(item[1])
            return tmpPath;
          return item[3]
    
    state = theQueue.pop();

  return []
      
def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  seenAlready = {}
  startState = problem.getStartState();
  state = [startState,"",1,[]];
  theQueue = util.PriorityQueue();
  seenAlready[state[0]] = "1"
  path = []

  if (problem.isGoalState(startState)) {
    return []
  }
  while(1):    
    children = problem.getSuccessors(state[0]);
    
    if (children):
      for item in children:
        if (item[0] not in seenAlready):
          tmpPath = list(state[3]);
          tmpPath.append(item[1])
          item = list(item);
          item.append(tmpPath)
          theQueue.push(item,problem.getCostOfActions(tmpPath))
          seenAlready[item[0]] = "1"
        if (problem.isGoalState(item[0])):
          return item[3]
    
    state = theQueue.pop();

  return []

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def manhattanHeuristic(state, problem=None):
  
  return manhattanDistance(state,problem.getGoalState());


def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  
  seenAlready = {}
  startState = problem.getStartState();
  state = [startState,"",1,[]];
  theQueue = util.PriorityQueue();
  seenAlready[state[0]] = "1"
  path = []

  if (problem.isGoalState(startState)) {
    return []
  }
  
  while(1):    
    children = problem.getSuccessors(state[0]);
    
    if (children):
      for item in children:
        if (item[0] not in seenAlready):
          tmpPath = list(state[3]);
          tmpPath.append(item[1])
          item = list(item);
          item.append(tmpPath)
          theQueue.push(item,problem.getCostOfActions(tmpPath)+heuristic(item[0],problem))
          seenAlready[item[0]] = "1"
        if (problem.isGoalState(item[0])):
          return item[3]
    
    state = theQueue.pop();

  return []
    
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch