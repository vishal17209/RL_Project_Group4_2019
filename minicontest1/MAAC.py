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
		self.agent_1 = ReinforcementAgent.__init__(self, **args)
		self.agent_2 = ReinforcementAgent.__init__(self, **args)

		self.action_values_1 = util.Counter()
		self.action_values_2 = util.Counter()

		self.policy_params_1 = np.random.rand(100, 1)
		self.policy_params_2 = np.random.rand(100, 1)

		self.replay_buffer = util.Counter() # Store as: (current state, next state, (action 1, action 2), (reward 1, reward 2))

	def getActionValue(self, agent, red_state, action_1, action_2):
		"""
		From big Q function
		"""
		raise NotImplementedError

	def getAction(self, params, observe):
		"""
		Uses policy parameters to return action, 
		"""

	def getPolicyParamUpdate(self, agent, params):
		"""
		Return policy parameter update 
		"""
		raise NotImplementedError

	def featureExtractor(self, state):
		"""
		Return reduced states for all agents in a tuple or list
		"""
		raise NotImplementedError

