# The AI

# Importing the libraries
import numpy as np
import random
import os             
import torch

# Tools need to implement neural net
import torch.nn as nn
# Functional package contains the uberloss to speed convergence
import torch.nn.functional as F 
# Optimiser to perform stochastic gradient decent
import torch.optim as optim 
# To convert from torch-tensors to a variable which contains the tensor and a gradient
import torch.autograd as autograd; from torch.autograd import Variable

# Creating the architecture of the neural network
class NeuralNetwork(object):
	""" The network will take inputs from the sensors and give us one of 
		3 possible actions by forward propagating our out input values to
		be backpropagated later to optimise our weights.
    """
	def __init__(self, input_neurons, neurons_output):
		# Inherits from nn.Module. Neurons_out is the number of possible actions
		super(NeuralNetwork, self).__init__()

		# Attatching our parameters as variables to the later created object
        self.input_neurons = input_neurons
        self.neurons_output = neurons_output

        ''' Creating synaptical connection between layers 
            input --> hidden --> output 
            REMEMBER TO MESS WITH HYPERPARAMETER(30) LATER '''
        self.input_layer = nn.Linear(self.input_neurons, 30)
        self.hidden_layer = nn.Linear(30, self.neurons_output)

    def forward_propagate(self, state):
    	# Applying a rectifyer activation function against our hypothesised inputs
    	activated_neurons = F.relu(self.input_layer(state))
    	q_val_action = self.hidden_layer(activated_neurons)

    	# Returning n number of actions the car can take 
    	return q_val_action

# Experience replay 
class ExperienceReplay(object):
	""" This is too prevent independencies/correlations between consecutive states from
		biasing our network by storing interdependent states into batch experiences and 
		then putting them into the network after. 

		This is done by taking a uniformly distributed and random selection of a batch 
		of experiences to learn from.

		This is where an experience is defined --> [s, a, s_prime, reward]...
		The sample batches are reshaped experiences being [[s1, s2, s3], [a1, a2, a3], [s_prime1, s_prime2, s_prime3]]
	"""

	''' Experiement with the batch capacity hyperparameter since less structured curves 
	    are more structured with smaller rolling windows (batch sizes).
	'''

	def __init__(self, capacity):
		''' Capacity dictates how many batch experiences can can be stored in our sliding
		    window '''
		self.capacity = capacity
		
		# This is the sliding window a list of batch experiences.
		self.memory = []

		# Pushes an experience batch to our memory's sliding window...
		def push(self, experience):
			self.memory.append(experience)
			if len(self.memory) > self.capacity:
				del self.memory[0]

		# We're batching up samples of experiences for experience replay
		def sample(self, batch_size):
			
			# Zipping corresponding batch features together as explained in docs
			samples = zip(*random.sample(self.memory, batch_size))

			''' Wrapping each of all our samples into a pytorch variable for backpropagation
			    and then reshaping these samples so that they're shaped with respect
			    to time. '''
			return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning
class DeepQNetwork(object):
	""" This is different to the neural network which just mapped out
	 	the network's computational graph. This is the model in action.
	 	our self.dqn_model can take values as that's how nn.Linear works.
	"""
	def __init__(self, input_neurons, neurons_output, gamma):

		""" Discount factor - it models the fact future reward is worth less than 
		immediate reward. """
		self.gamma = gamma

		# Let's us track performance by storing mean of last n rewards (should increase)
		self.reward_window = []

		self.dqn_model = NeuralNetwork(input_neurons, neurons_output)
		self.memory = ExperienceReplay(100000)
		
		# Optimiser for stochastic gradient decent
		self.optimiser = optim.Adam(self.dqn_model.parameters(), lr=0.001)

		''' We store the last state as a tensor of size n and then we unsqueeze is 
			to get a fake dimension as a tensor of size 1 x n which means that we 
			can use treat it as a batch '''
		self.last_state = torch.Tensor(input_neurons).unsqueeze(0)

		# Either index 0, 1 or 2 referring to rotations that can be made. [0, 20, -20]
		self.last_action = 0

		# Initialising as 0
		self.last_reward = 0