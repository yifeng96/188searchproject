######################
# ANALYSIS QUESTIONS #
######################

# Change these default values to obtain the specified policies through
# value iteration.

def question2():
  answerDiscount = 0.9
  answerNoise = 0.0
  return answerDiscount, answerNoise

def question3a():
  answerDiscount = 0.9
  answerNoise = 0.2
  answerLivingReward = -3.0
  return answerDiscount, answerNoise, answerLivingReward
  # If not possible, return 'NOT POSSIBLE'

def question3b():
  answerDiscount = 0.2
  answerNoise = 0.2
  answerLivingReward = 0.0
  return answerDiscount, answerNoise, answerLivingReward
  # If not possible, return 'NOT POSSIBLE'

def question3c():
  answerDiscount = 0.9
  answerNoise = 0.2
  answerLivingReward = -1.0
  return answerDiscount, answerNoise, answerLivingReward
  # If not possible, return 'NOT POSSIBLE'

def question3d():
  answerDiscount = 0.9
  answerNoise = 0.2
  answerLivingReward = 0.0
  return answerDiscount, answerNoise, answerLivingReward
  # If not possible, return 'NOT POSSIBLE'

def question3e():
  answerDiscount = 0.9
  answerNoise = 0.2
  answerLivingReward = 100
  return answerDiscount, answerNoise, answerLivingReward
  # If not possible, return 'NOT POSSIBLE'

def question6():
  answerEpsilon = None
  answerLearningRate = None
  return 'NOT POSSIBLE' # No matter what learning rate (alpha) is, you need to have random actions
  # "take you" five tiles to the East consecutively. If you have epsilon = 1 (i.e ignoring learning),
  # then the probability of that is VERY small (0.25 * (0.25 * (0.25 * (0.25 * (0.25))))) = 0.00098,
  # i.e 0.098% chance. If you throw learning into the fray, then the agent will ALWAYS want to move
  # West, because it's reward is guaranteed to be 1, so the agent would have to rely on its epsilon
  # to generate enough "randomness" to throw it to the East five tiles. So, the best case probability
  # is when epsilon = 1, i.e 0.098%.  
  return answerEpsilon, answerLearningRate
  # If not possible, return 'NOT POSSIBLE'
  # return not possible
  
if __name__ == '__main__':
  print 'Answers to analysis questions:'
  import analysis
  for q in [q for q in dir(analysis) if q.startswith('question')]:
    response = getattr(analysis, q)()
    print '  Question %s:\t%s' % (q, str(response))
