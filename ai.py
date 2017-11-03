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
# To convert from tensors to a variable which contains a gradient
import torch.autograd as autograd
from torch.autograd import variable

# Creating the architecture of the neural network

class Neural_network(nn.Module): # Inheriting from this module parent class

    # Creating the architecture of the neural network
    def __init__(self, input_neurons, output_neurons): 
        super(Neural_network, self).__init__() # Inherits from module
        # input_neurons is the number of dimensions of inputs that our model takes from it's environment
        # Rotation, Rotations negative, 3 sensors
        # output_neurons: up, left or right        

        # Attaching these dimension variables to our object
        self.input_neurons = input_neurons 
        self.output_neurons = output_neurons

        # Mapping out full connections between layers. 30 = hidden layer nodes
        self.full_connection1 = nn.Linear(self, input_neurons, 30)
        self.full_connection2 = nn.Linear(self, 30, output_neurons)

        # Returns the q-values for each possible action depending on the state
        def forward_propagate(self, state):
            # Activating the hidden neurons

            # Using the rectifier function "relu" to activate the hidden_neurons
            hidden_neurons = F.relu(self.full_connection1(state)) # In full_conn... we gave an argument of out input state to go from our input -> hidden_neurons
            
            # Hidden neurons are activated...

            # From hidden, neurons we output q_values to output neurons
            q_values = self.full_connection2(hidden_neurons)
            return q_values

        # Emplementing experience replay

