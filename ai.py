# AI for Self Driving Car

# Importing the libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the neural network

class Neural_net(nn.Module): # We're inheriting from all the tools of the neural net module
    
    # Any variable below with self.VAR_NAME shows that the variable is an abject of this class
    
    def __init__ (self, input_neurons, np_actions): 
        # Input neurons is the number of inputs that our model takes from it's environment
        # np_actions are the output neurons: up, left or right

        super(Neural_net, self).__init__() # This lets us use all the tools of nn.module
        self.input_neurons = input_neurons # Attatching the input size to the object
        self.np_actions = np_actions       # Attatching the output neurons to the object
        
        # Full_connections_between(input, hidden layer)
        self.full_connection1 = nn.Linear(self.input_neurons, 30) 

        # Full_connections_between(hidden layer, output layer)
        self.full_connection2 = nn.Linear(30, np_actions)