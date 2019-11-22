from game import Agent
from learningAgents import ReinforcementAgent
from searchProblems import PositionSearchProblem

import util
import time
import search
import numpy as np

class MultiActorCritic(ReinforcementAgent):

	'''
		Action values are calculated using Approx Q
		Use feature extraction from ../reinforcement
	'''
	def __init__(self, **args):
		ReinforcementAgent.__init__(self, **args)
		self.n = 2
		self.parameters = [util.Counter() for i in range(self.n)]
		self.policies = [util.Counter() for i in range(self.n)]
		self.action_values = [util.Counter() for i in range(self.n)]

	def thisIsIT(self, state):
		pacmanPosition = state.getPacmanPosition()
		grid = state.data.ToList()
		height, width = state.data.layout.height, state.data.layout.width
		new_state = grid.data[max(0, pacmanPosition[0]-3):min(height-1, pacmanPosition[0]+3)][max(0, pacmanPosition[1]-3):min(width-1, pacmanPosition[1]+3)]

		# return new_state
		return state

	def feature_extractor(self, state, index):
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

		print(foods)
		print(walls)


	def gradient(self, state, acitons):
		raise NotImplementedError
	
state = "  %%%%%%%\n% P   %\n% %%% %\n% %.  %\n% %%% %\n%. G  %\n%%%%%%%"
