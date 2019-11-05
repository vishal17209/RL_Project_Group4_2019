# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
	"""
	  Q-Learning Agent

	  Functions you should fill in:
		- computeValueFromQValues
		- computeActionFromQValues
		- getQValue
		- getAction
		- update

	  Instance variables you have access to
		- self.epsilon (exploration prob)
		- self.alpha (learning rate)
		- self.discount (discount rate)

	  Functions you should use
		- self.getLegalActions(state)
		  which returns legal actions for a state
	"""
	def __init__(self, **args):
		"You can initialize Q-values here..."
		ReinforcementAgent.__init__(self, **args)

		"*** YOUR CODE HERE ***"
		self.action_values = util.Counter()

	def thisIsIT(self, state):
		pacmanPosition = state.getPacmanPosition()
		grid = state.data.ToList()
		height, width = state.data.layout.height, state.data.layout.width
		new_state = grid.data[max(0, pacmanPosition[0]-3):min(height-1, pacmanPosition[0]+3)][max(0, pacmanPosition[1]-3):min(width-1, pacmanPosition[1]+3)]

		return new_state


	def getQValue(self, state, action):
		"""
		  Returns Q(state,action)
		  Should return 0.0 if we have never seen a state
		  or the Q node value otherwise
		"""
		"*** YOUR CODE HERE ***"
		self.action_values[(state, action)]

	def computeValueFromQValues(self, state, compressed_state):
		"""
		  Returns max_action Q(state,action)
		  where the max is over legal actions.  Note that if
		  there are no legal actions, which is the case at the
		  terminal state, you should return a value of 0.0.
		"""
		"*** YOUR CODE HERE ***"
		values = [self.action_values[(compressed_state, action)] for action in self.getLegalActions(state)]
		if(len(values) == 0):
			return 0.0
		else:
			return max(values)

	def computeActionFromQValues(self, state, compressed_state):
		"""
		  Compute the best action to take in a state.  Note that if there
		  are no legal actions, which is the case at the terminal state,
		  you should return None.
		"""
		"*** YOUR CODE HERE ***"
		values = [self.action_values[(compressed_state, action)] for action in self.getLegalActions(state)]
		opt_value = 0
		if(len(values) == 0):
			opt_value = 0.0
		else:
			opt_value = max(values)
		
		for action in self.getLegalActions(state):
			if(self.action_values[(compressed_state, action)] == opt_value):
				return action

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
		compressed_state = str(self.thisIsIT(state))
		legalActions = self.getLegalActions(state)
		action = None
		"*** YOUR CODE HERE ***"
		if(len(legalActions) != 0):
			if(random.random() < self.epsilon):
				return random.choice(legalActions)
			return self.computeActionFromQValues(state, compressed_state)
		return action

	def update(self, state, action, nextState, reward):
		"""
		  The parent class calls this to observe a
		  state = action => nextState and reward transition.
		  You should do your Q-Value update here

		  NOTE: You should never call this function,
		  it will be called on your behalf
		"""
		"*** YOUR CODE HERE ***"
		compressed_state = str(self.thisIsIT(state))
		compressed_nextState = str(self.thisIsIT(nextState))
		self.action_values[(compressed_state, action)] += self.alpha*(reward + self.discount*self.computeValueFromQValues(nextState, compressed_nextState) - self.action_values[(compressed_state, action)])

	# def getPolicy(self, state):
	# 	return self.computeActionFromQValues(state)

	# def getValue(self, state):
	# 	return self.computeValueFromQValues(state)


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
		self.index = 0  # This is always Pacman
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
		self.weights = util.Counter()

	def thisIsIT(self, state):
		pacmanPosition = state.getPacmanPosition()
		grid = state.data.ToList()
		height, width = state.data.layout.height, state.data.layout.width
		new_state = grid.data[max(0, pacmanPosition[0]-3):min(height-1, pacmanPosition[0]+3)][max(0, pacmanPosition[1]-3):min(width-1, pacmanPosition[1]+3)]

		return new_state

	def getWeights(self):
		return self.weights

	def getQValue(self, state, action):
		"""
		  Should return Q(state,action) = w * featureVector
		  where * is the dotProduct operator
		"""
		"*** YOUR CODE HERE ***"
		compressed_state = str(self.thisIsIT(state))
		feature_vector = self.featExtractor.getFeatures(compressed_state, action)
		return self.weights*feature_vector

	def update(self, state, action, nextState, reward):
		"""
		   Should update your weights based on transition
		"""
		"*** YOUR CODE HERE ***"
		compressed_state = str(self.thisIsIT(state))
		diff = reward + self.discount*self.computeValueFromQValues(state, compressed_state) - self.getQValue(state, action)
		feature_vector = self.featExtractor.getFeatures(compressed_state, action)
		for key in feature_vector.keys():
			self.weights[key] += self.alpha*diff*feature_vector[key]

	def final(self, state):
		"Called at the end of each game."
		# call the super-class final method
		PacmanQAgent.final(self, state)

		# did we finish training?
		if self.episodesSoFar == self.numTraining:
			# you might want to print your weights here for debugging
			"*** YOUR CODE HERE ***"
			pass
