# myAgents.py
# ---------------
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

from game import Agent
from searchProblems import PositionSearchProblem
from learningAgents import ReinforcementAgent
from pacman import PacmanRules

import random,math
import util
import time
import search
import copy

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='QLearningAgent'):
	return [eval(agent)(index=i) for i in range(num_pacmen)]

# class MyAgent(Agent):
#     """
#     Implementation of your agent.
#     """

#     def getAction(self, state):
#         """
#         Returns the next action the agent will take
#         """

#         "*** YOUR CODE HERE ***"

#         raise NotImplementedError()

#     def initialize(self):
#         """
#         Intialize anything you want to here. This function is called
#         when the agent is first created. If you don't need to use it, then
#         leave it blank
#         """

#         "*** YOUR CODE HERE"

#         raise NotImplementedError()

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)


        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        fringe = util.Queue()
        visited = []        # List of already visited nodes
        action_list = []    # List of actions taken to get to the current node
        total_cost = 0      # Cost to get to the current node
        initial = problem.getStartState()   # Starting state of the problem

        fringe.push((initial, action_list))

        while fringe:
            node, actions = fringe.pop()
            if not node in visited:
                visited.append(node)
                if problem.isGoalState(node):
                    return actions
                successors = problem.getSuccessors(node)
                for successor in successors:
                    coordinate, direction, cost = successor
                    fringe.push((coordinate, actions + [direction]))

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.food[x][y]


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
	def __init__(self, epsilon=0.3,gamma=0.9,alpha=1, numTraining=100, **args):

		"You can initialize Q-values here..."
		args['epsilon'] = epsilon
		args['gamma'] = gamma
		args['alpha'] = alpha
		args['numTraining'] = numTraining
		ReinforcementAgent.__init__(self, **args)
		print(self.epsilon, self.alpha,self.numTraining) #whoami

		"*** YOUR CODE HERE ***"
		self.action_values = util.Counter()

	def thisIsIT(self, state):
		pacmanPosition = state.getPacmanPosition(self.index)
		grid = str(state.data.ToList())
		grid=grid.split("\n")
		height, width = state.data.layout.height, state.data.layout.width   

		#for vision = (2vision+1)x(2vision+1) #whoami
		vision=2
		new_state = grid[max(0, (height-1-pacmanPosition[1])-vision):min(height, (height-pacmanPosition[1])+vision)]
		for lul in range(len(new_state)):
			new_state[lul]=new_state[lul][max(0, pacmanPosition[0]-vision):min(len(grid[0]), 1+pacmanPosition[0]+vision)]
		new_state="\n".join(new_state)
		
		return new_state
		# return state.getPacmanState( self.index ) #whoami
		# return pacmanPosition #whoami

	def getQValue(self, state, action):
		"""
		  Returns Q(state,action)
		  Should return 0.0 if we have never seen a statestate.getPacmanPosition(self.index)
		  or the Q node value otherwise
		"""
		"*** YOUR CODE HERE ***"
		if((state,action) not in action_values):
			return 0.0 #whoami
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
		compressed_state = self.thisIsIT(state.deepCopy())

		#whoami for testing compressed state formation
		# if(self.index==0):
		# 	height, width = state.data.layout.height, state.data.layout.width
		# 	grid = str(state.data.ToList())
		# 	print("state",grid) #whoami
		# 	pacmanPosition=state.getPacmanPosition(self.index)
		# 	print(pacmanPosition)
		# 	print("row clip",max(0, (height-1-pacmanPosition[1])-1),min(height, (height-pacmanPosition[1])+1))
		# 	print("column clip",max(0, pacmanPosition[0]-1),min(len(grid[0]), pacmanPosition[0]+2),grid[0])
		# 	print("new_state", self.index) 
		# 	print(compressed_state) 

		legActions = self.getLegalActions(state)
		# print(legActions)
		action = None
		"*** YOUR CODE HERE ***"
		print(self.numTraining - self.episodesSoFar,"trainingleft",self.epsilon, "epsilon") #whoami
		
		if(len(legActions) != 0):
			if(random.random() < self.epsilon):
				action = random.choice(legActions)
			else:
				action = self.computeActionFromQValues(state, compressed_state)

		self.doAction(state.deepCopy(),action)#whoami

		if(self.index==0): #being recorded only for first pacman right now
			f = open("actions.txt", "a")
			f.write("get action\n")
			f.write(str(compressed_state)+"\n");f.write(str(legActions)+" "+str(action)+"\n")
			f.close()

		# assert("P" in  compressed_state), compressed_state #whoami

		return action #whoami

	def update(self, state, action, nextState, reward):
		"""
		  The parent class calls this to observe a
		  state = action => nextState and reward transition.
		  You should do your Q-Value update here

		  NOTE: You should never call this function,
		  it will be called on your behalf
		"""
		"*** YOUR CODE HERE ***"
		print(self.numTraining - self.episodesSoFar,"trainingleft",self.alpha, "alpha",self.discount, "discount") #whoami
		
		compressed_state = self.thisIsIT(state.deepCopy())
		compressed_nextState = self.thisIsIT(nextState.deepCopy())

		if(self.index==0): #being recorded only for first pacman right now
			f = open("actions.txt", "a")
			f.write("updation\n")
			f.write("state\n")
			f.write(str(state.data.ToList())+"\n")
			f.write("compstate\n")
			f.write(str(compressed_state)+"\n");
			f.write(str(action)+"\n")
			f.write("nextstate\n")
			f.write(str(nextState.data.ToList())+"\n")
			f.write("compnextstate\n")
			f.write(str(compressed_nextState)+"\n")
			f.write(str(reward)+" "+str(self.discount*self.computeValueFromQValues(nextState, compressed_nextState))+" "+str(self.action_values[(compressed_state, action)])+" "+str(self.alpha*(reward + self.discount*self.computeValueFromQValues(nextState, compressed_nextState) - self.action_values[(compressed_state, action)]))+"\n")
			f.close()#whoami

		# assert("P" in compressed_nextState and "P" in compressed_state), compressed_state+"\n\n"+compressed_nextState+"\n"+str(nextState.getLegalActions(self.index)) #whoami

		self.action_values[(compressed_state, action)] += self.alpha*(reward + self.discount*self.computeValueFromQValues(nextState, compressed_nextState) - self.action_values[(compressed_state, action)])


	#whoami ignore them for now
	def getPolicy(self, state):
		return self.computeActionFromQValues(state)

	def getValue(self, state):
		return self.computeValueFromQValues(state)









# class MyAgent(QLearningAgent):
# 	"Exactly the same as QLearningAgent, but with different default parameters"

# 	def __init__(self, epsilon=0.5,gamma=0.9,alpha=0.3, numTraining=0, **args):
# 		"""
# 		These default parameters can be changed from the pacman.py command line.
# 		For example, to change the exploration rate, try:
# 			python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

# 		alpha    - learning rate
# 		epsilon  - exploration rate
# 		gamma    - discount factor
# 		numTraining - number of training episodes, i.e. no learning after these many episodes
# 		"""

# 	def getAction(self, state):
# 		"""
# 		Simply calls the getAction method of QLearningAgent and then
# 		informs parent of action for Pacman.  Do not change or remove this
# 		method.
# 		"""
# 		action = QLearningAgent.getAction(self,state)
# 		self.doAction(state,action)
# 		return action




#state generalization required, reward incentivization required

#in the state(valid actions remain intact), but the description priority remains:
#power up overlaps all, ghost overlaps pacman and pacman overlaps scared ghost 
#also, power up reward added, pellet reward and time penalty changed, and scared ghost visibility added
