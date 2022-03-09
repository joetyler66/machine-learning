"""
perceptron.py
"""

#Libraries used
import numpy as np 
import random

class Network: #(object):

  def __init__(self, sizes):
    '''
    initialise network with sizes[0] inputs, sizes[i] neurons in the ith hidden layer and sizes[end]
    weights and biases are initially randomized with a normal distribution
    '''
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1],sizes[1:])]
