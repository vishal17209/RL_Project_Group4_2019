from game import Agent
from searchProblems import PositionSearchProblem
from learningAgents import ReinforcementAgent
from pacman import PacmanRules

import random,math
import util
import time
import search
import copy
import numpy as np

class MultiAgentActorCritic(ReinforcementAgent):
	"""
	Default 2 pacmen are spawned
	"""
	def __init__(self, **args):	
		args['epsilon'] = epsilon
		args['gamma'] = gamma
		args['alpha'] = alpha
		args['numTraining'] = numTraining

		self.lr = 1e-4

		self.num_agents = 2

		self.agent_1 = ReinforcementAgent.__init__(self, **args)
		self.agent_2 = ReinforcementAgent.__init__(self, **args)

		self.action_values_1 = util.Counter()
		self.action_values_2 = util.Counter()
		self.action_values = [self.action_values_1, self.action_values_2]

		self.policy_params_1 = np.random.rand(100, 1) # Fix this!
		self.policy_params_2 = np.random.rand(100, 1) # Fix this!
		self.policy_params = [self.policy_params_1, self.policy_params_2]

		self.replay_buffer = util.Counter() # Store as: ((current state, next state), (action 1, action 2), (reward 1, reward 2))

	def getActionValue(self, observes, actions, action_values):
		"""
		From big Q function, associate value with agent observation.
		actions = (action_1, action_2)
		observes = (observe_1, observe_2)
		"""
		# raise NotImplementedError
		return action_values[(observes, actions)] 

	def getAction(self, agent_idx, observe, policy_params, state, action_values):
		"""
		Uses policy parameters to return actions
		"""
		# raise NotImplementedError
		legal_actions = state.getLegalActions(state, agent_idx) # Make sure that getLegalActions is returning legal actions for our agent.
		h_values = [(np.dot(np.concatenate((observe, action), axis=None), policy_params), action) for action in legal_actions] # Not bothering with softmax here...

		return max(h_values)[1]

	def getPolicyParamUpdate(self, curr_state, curr_actions, rewards, next_state):
		"""
		Return policy parameter update 
		"""
		# raise NotImplementedError
		# Sample from replay buffer ...
		# states, actions, rewards,  = np.random.randint(len(self.replay_buffer))

		curr_observes = self.featureExtractor(curr_state) # for the corresponding agent index.
		f = np.exp(np.dot(np.concatenate((curr_observes, curr_actions[self.index]), axis=None), self.policy_params))
		f_sum = 0
		for action in curr_state.getLegalActions(self.index):
			f_sum += np.exp(np.dot(np.concatenate((curr_observes, action), axis=None), self.policy_params))

		x = f/f_sum
		return self.lr*(1-x)

	def getActionValueUpdate(self, agent_idx, curr_state, curr_actions, next_state, next_actions, reward):
		"""
		Get ActionValue update (just add to real action value)
		actions: must be tuple of all actions (indexed)
		"""
		curr_observe = self.featureExtractor(curr_state)
		next_observe = self.featureExtractor(next_state)
		next_actions = []
		for i in range(len(curr_actions)):
			next_actions.append(self.getAction(self.index, next_observe, self.policy_params, state, self.action_values))

		next_actions = tuple(next_actions)
		self.action_values[(curr_observe, curr_actions)] += self.alpha*(reward + self.discount*self.getActionValue(next_observe, next_actions) - self.getActionValue(curr_observe, curr_actions))

	def featureExtractor(self, state):
		"""
		Return reduced states for all agents in a tuple or list
		"""
		raise NotImplementedError
		
		offset = 3
		compressed_state = self.thisIsIT(state)
		#remove only the line below
		compressed_state = compressed_state.ToList()
		food = state.getFood()
		walls = state.getWalls()
		ghostPositions = state.getGhostPositions()
		isscared = state.getGhostStates()[0].scaredTimer > 0
		pacmanPosition = [state.getPacmanPosition(i) for i in range(self.n)]
		#state = (encoded food positions, encoded pacman position, encoded wall positions, encoded capsule positions, ghostScared)
		i_start = pacmanPosition[index] - offset
		i_end = pacmanPosition[index] + offset
		j_start = pacmanPosition[index] - offset
		j_end = pacmanPosition[index] + offset 
		
		foods = []
		walls = []
		for i in range(len(food)):
			for j in range(len(food[0])):
				if(food[i][j]):
					foods.append((i,j))
				if(walls[i][j]):
					walls.append(i,j)

		return state
