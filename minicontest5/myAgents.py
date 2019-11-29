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
import numpy as np

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='MultiAgentActorCritic'):
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
	def __init__(self, epsilon=0.1,gamma=0.9,alpha=1, numTraining=40000, **args):

		"You can initialize Q-values here..."
		args['epsilon'] = epsilon
		args['gamma'] = gamma
		args['alpha'] = alpha
		args['numTraining'] = numTraining
		ReinforcementAgent.__init__(self, **args)
		print(self.epsilon, self.alpha,self.numTraining) #whoami

		"*** YOUR CODE HERE ***"
		self.action_values = util.Counter()
		self.action_values_num = util.Counter()

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
		# print(self.numTraining - self.episodesSoFar,"trainingleft",self.epsilon, "epsilon") #whoami
		
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
		# print(self.numTraining - self.episodesSoFar,"trainingleft",self.alpha, "alpha",self.discount, "discount") #whoami
		
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

		#whoami
		self.action_values_num[(compressed_state, action)]+=1
		self.action_values[(compressed_state, action)] += self.alpha*(reward + self.discount*self.computeValueFromQValues(nextState, compressed_nextState) - self.action_values[(compressed_state, action)])/max(1,self.action_values_num[(compressed_state, action)])
		# self.action_values[(compressed_state, action)] += self.alpha*(reward + self.discount*self.computeValueFromQValues(nextState, compressed_nextState) - self.action_values[(compressed_state, action)])


	#whoami ignore them for now
	def getPolicy(self, state):
		return self.computeActionFromQValues(state)

	def getValue(self, state):
		return self.computeValueFromQValues(state)










class MultiAgentActorCritic(ReinforcementAgent):
	"""
	Default 2 pacmen are spawned
	"""
	def __init__(self, epsilon=0.2,gamma=0.9,alpha=1, numTraining=500, **args):	


		"You can initialize Q-values here..."
		args['epsilon'] = epsilon
		args['gamma'] = gamma
		args['alpha'] = alpha
		args['numTraining'] = numTraining
		ReinforcementAgent.__init__(self, **args)
		print(self.epsilon, self.alpha,self.numTraining) #whoami

		"*** YOUR CODE HERE ***"
		self.action_values = util.Counter()
		self.action_values_num = util.Counter()

		self.vision = 2 #only set this whenever vision is to be adjusted

		self.blockvision=(2*self.vision)+1

		self.lr = 10**(-3)

		self.policy_params = np.random.randn((self.blockvision*self.blockvision*7)+5) # equal to feature vector size

		#whoami
		self.policy_params_change = np.random.randn((self.blockvision*self.blockvision*7)+5) # equal to feature vector size

	
	def thisIsIT(self, state):
		pacmanPosition = state.getPacmanPosition(self.index)
		grid = str(state.data.ToList())
		grid=grid.split("\n")
		height, width = state.data.layout.height, state.data.layout.width   

		#for vision = (2vision+1)x(2vision+1) #whoami
		vision=self.vision
		new_state = grid[max(0, (height-1-pacmanPosition[1])-vision):min(height, (height-pacmanPosition[1])+vision)]
		for lul in range(len(new_state)):
			new_state[lul]=new_state[lul][max(0, pacmanPosition[0]-vision):min(len(grid[0]), 1+pacmanPosition[0]+vision)]
		new_state="\n".join(new_state)
		
		return new_state
		# return state.getPacmanState( self.index ) #whoami
		# return pacmanPosition #whoami



	def getActionValue(self, observes, actions): #unused for now. But can be used if action values needed outside the class
		"""
		From big Q function, associate value with agent observation.
		actions = (action_1, action_2)
		observes = (observe_1, observe_2)
		"""
		# raise NotImplementedError
		return self.action_values[(observes, actions)] 

	
	def getAction(self, state):
		"""
		Uses policy parameters to return actions
		"""
		# raise NotImplementedError
		legal_actions = state.getLegalActions(self.index) # Make sure that getLegalActions is returning legal actions for our agent.
		

		observe=self.featureExtractor(state.deepCopy())

		h_values = [] # Not bothering with softmax here...
		for action in legal_actions:
			hot=np.zeros(5)
			if(action=="North"):
				hot[0]=1
			elif(action=="East"):
				hot[1]=1
			elif(action=="South"):
				hot[2]=1
			elif(action=="West"):
				hot[3]=1
			elif(action=="Stop"):
				hot[4]=1
			else:
				hot[5]=1

			h_values.append( (np.dot(np.concatenate((observe, hot), axis=None), self.policy_params), action) )

		
		f_sum = 0; temp1=[]; temp2=[] # bothering with softmax here...  lol
		for action in state.getLegalActions(self.index):
			hot=np.zeros(5)
			if(action=="North"):
				hot[0]=1
			elif(action=="East"):
				hot[1]=1
			elif(action=="South"):
				hot[2]=1
			elif(action=="West"):
				hot[3]=1
			elif(action=="Stop"):
				hot[4]=1
			else:
				hot[5]=1
			
			val = np.exp(np.dot(np.concatenate((observe, hot), axis=None), self.policy_params))
			f_sum += float(val) 
			temp1.append(val);temp2.append(action)

		for i in range(len(temp1)):
			temp1[i]=temp1[i]/f_sum
		
		print(temp1, temp2, self.index) #for seeing prob distribution over the possible actions
		prob_dist=[];acc=0
		for i in range(len(temp2)):
			acc+=temp1[i]
			prob_dist.append(acc)
		
		assert(abs(prob_dist[-1]-1)<10**(-3)), "prob_dist sum varying too much from 1 in the end"+str(prob_dist)

		if(self.isInTraining()):
			tick=random.random()
			if(tick<self.epsilon):
				return random.choice(temp2)
			else:
				num=random.random()
				
				for i in range(1,len(prob_dist)):
					if(num<=prob_dist[i] and num>prob_dist[i-1]):
						return temp2[i]
				return temp2[0]
		
		else:
			tick=random.random()
			if(tick<self.epsilon):
				return random.choice(temp2)
			else:
				num=random.random()
				
				for i in range(1,len(prob_dist)):
					if(num<=prob_dist[i] and num>prob_dist[i-1]):
						return temp2[i]
				return temp2[0]
			
			# return max(h_values)[1]	
	
	

	def getPolicyParamUpdate(self, curr_state, curr_actions, rewards, next_state, it):
		"""
		Return policy parameter update 
		"""
		# raise NotImplementedError
		# Sample from replay buffer ...
		# states, actions, rewards,  = np.random.randint(len(self.replay_buffer))

		curr_observe = self.featureExtractor(curr_state.deepCopy()) # for the corresponding agent index.
		
		f_sum = 0; temp=np.zeros((self.blockvision*self.blockvision*7)+5)
		for action in curr_state.getLegalActions(self.index):
			hot=np.zeros(5)
			if(action=="North"):
				hot[0]=1
			elif(action=="East"):
				hot[1]=1
			elif(action=="South"):
				hot[2]=1
			elif(action=="West"):
				hot[3]=1
			elif(action=="Stop"):
				hot[4]=1
			else:
				hot[5]=1
			
			val = np.exp(np.dot(np.concatenate((curr_observe, hot), axis=None), self.policy_params))
			f_sum += val 
			temp += val*(np.concatenate((curr_observe, hot), axis=None))



		hot=np.zeros(5)
		if(curr_actions[self.index]=="North"):
			hot[0]=1
		elif(curr_actions[self.index]=="East"):
			hot[1]=1
		elif(curr_actions[self.index]=="South"):
			hot[2]=1
		elif(curr_actions[self.index]=="West"):
			hot[3]=1
		elif(curr_actions[self.index]=="Stop"):
			hot[4]=1
		else:
			hot[5]=1

		# self.policy_params += self.lr*self.action_values[(tuple(curr_observe),tuple(curr_actions))]*(np.concatenate((curr_observe, hot), axis=0) - (temp/f_sum))
		if(it==0):
			self.policy_params_change= np.zeros((self.blockvision*self.blockvision*7)+5)
		self.policy_params_change += self.lr*self.action_values[(tuple(curr_observe),tuple(curr_actions))]*(np.concatenate((curr_observe, hot), axis=0) - (temp/f_sum))

	
	def update_the_params(self, batch_size):
		self.policy_params += (self.policy_params_change)/batch_size		
		print(self.numTraining - self.episodesSoFar,"trainingleft",self.alpha, "alpha",self.discount, "discount") #whoami

	
	def getActionValueUpdate(self, curr_state, curr_actions, next_state, reward):
		"""
		Get ActionValue update (just add to real action value)
		actions: must be tuple of all actions (indexed)
		"""
		curr_observe = self.featureExtractor(curr_state.deepCopy())
		next_observe = self.featureExtractor(next_state.deepCopy())
		next_actions = []
		for i in range(len(curr_actions)):
			next_actions.append(self.getAction( next_state ))

		next_actions = tuple(next_actions)
		curr_actions = tuple(curr_actions)
		curr_observe = tuple(curr_observe)
		next_observe = tuple(next_observe)
		self.action_values_num[(curr_observe, curr_actions)]+=1
		self.action_values[(curr_observe, curr_actions)] += self.alpha*(reward[self.index] + self.discount*self.action_values[(next_observe, next_actions)] - self.action_values[(curr_observe, curr_actions)])/max(1,self.action_values_num[(curr_observe, curr_actions)])



	def featureExtractor(self, state):
		"""
		Return reduced states for all agents in a tuple or list
		"""
		
		compressed_state = self.thisIsIT(state.deepCopy())
		
		#7 things from a state
		pacman=np.zeros(self.blockvision*self.blockvision)
		wall=np.zeros(self.blockvision*self.blockvision)
		space=np.zeros(self.blockvision*self.blockvision)
		pellets=np.zeros(self.blockvision*self.blockvision)
		ghosts=np.zeros(self.blockvision*self.blockvision)
		scared=np.zeros(self.blockvision*self.blockvision)
		powerup=np.zeros(self.blockvision*self.blockvision)

		it=0
		for i in compressed_state:
			if(i=="A"):
				scared[it]=1
				it+=1
			elif(i=="G"):
				ghosts[it]=1
				it+=1
			elif(i=="P"):
				pacman[it]=1
				it+=1
			elif(i=="."):
				pellets[it]=1
				it+=1
			elif(i=="o"):
				powerup[it]=1
				it+=1
			elif(i==" "):
				space[it]=1
				it+=1
			elif(i=="%"):
				wall[it]=1
				it+=1
			else:
				pass

		return np.concatenate((scared, ghosts, pacman, pellets, powerup, space, wall), axis=None)		


















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
