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
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the neural network
class NeuralNetwork(nn.Module):
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
		self.input_layer = nn.Linear(input_neurons, 30)
		self.hidden_layer = nn.Linear(30, neurons_output)

	def forward(self, state):
		x = F.relu(self.input_layer(state))
		q_values = self.hidden_layer(x)
		return q_values

# Experience replay 
class ExperienceReplay(object):
	""" This is to prevent independencies/correlations between consecutive states from
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
class DeepQNetwork():
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

	def select_action(self, input_state):
		''' Chose a softmax because it's based on multinomial probability and
		so it will allow us to explore new options as opposed to an argmax. 

		We also don't need to differentiate because probabilities is based
		on an input value and it would make no sense to partially differentiate 
		loss with respect to an input value. So volatile=True to leave out the
		gradient. 

		Our temperature "T" dictates the probability (or how sure) that our model
		will settle for the final q_value. Our car will behave less insect like and
		more intelligently because the higher out temperature, the higher the prob
		of the winning q_value making the car seem more sure of where it's going. '''
		probabilities = F.softmax(self.dqn_model(Variable(input_state, volatile=True)) * 100) # T=100

		# Taking a random draw of the probabilities distribution
		action = probabilities.multinomial()

		# Getting actual action
		return action.data[0,0]

	# Our paramenters model a transition of the markov decision process
	def learn(self, batch_state, batch_next_state, batch_reward, batch_action):

		''' Obtaining Q-Values as outputs from putting batch_state through our deep q
		network and then we define outputs by the dimensions of batch_action. '''
		outputs = self.dqn_model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)

		# Returning only the maximum Q values of the next state for Q-learning formula
		# Format of next_outputs = s, a
		# Maximum of the q_vals of next state according to actions...
		next_outputs = self.dqn_model(batch_next_state).detach().max(1)[0]

		# Bellman's equation
		target = self.gamma * next_outputs + batch_reward

		# Our loss for gradient decent 
		td_loss = F.smooth_l1_loss(outputs, target)

		# Deleting gradients for new ones in forward propagation
		self.optimiser.zero_grad()

		# Back propagating the error in the network 
		td_loss.backward(retain_variables = True)

		# Updating weights
		self.optimiser.step()

	# Making an update to the rewards and signals for the car from the car's sensors
	def update(self, reward, new_signal):

		''' We'll input the last_signal from map which shows the state as orientation 
		and the 3 car signals '''
		new_state = torch.Tensor(new_signal).float().unsqueeze(0) # st + 1

		# Adding a transition to our memory: [st, st+1, a, r] 
		self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))

		# Play a new action after reaching a state
		action = self.select_action(new_state)

		if len(self.memory.memory) > 100:
			batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
			self.learn(batch_state, batch_next_state, batch_reward, batch_action)

		# Updating last_features to the features we just recalcualated
		self.last_action = action
		self.last_state = new_state
		self.last_reward = reward

		# Updating the reward window to track performance
		self.reward_window.append(reward)

		# Keeping the window of a fixed size
		if len(self.reward_window) > 1000:
			del self.reward_window[0]
		return action

	# Gives us the mean of the rewards in the sliding window 
	def score(self):
		return sum(self.reward_window)/len(self.reward_window + [1]) # Making sure len([rewardwindow]) > 1

	# Allows us to save models (last weights) & (last optimiser) into long term memory for reuse.
	def save(self):

		''' Saves the parmeter of our model as a corresponding value to our state dict
		key '''
		torch.save({'state_dict': self.dqn_model.state_dict(),
		'optimiser':self.optimiser.state_dict,
		}, 'lastModelAndOptim.pth')

	def load(self):

		# Checking if file exists
		if os.path.isfile('lastModelAndOptim.pth'):
			print('File found, loading last model...')
			last_model = torch.load('lastModelAndOptim.pth')

			# Loading last_model
			self.dqn_model.load_state_dict(last_model['state_dict'])
			# Loading last_optimiser
			self.dqn_model.load_state_dict(last_model['optimiser'])
			print('Loaded.')

		else:
			print("A saved model checkpoint isn't found...")
