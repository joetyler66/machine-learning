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
  
  def feedforward(self, a):
    """Calculates output of network given input a
    Output of each neuron is sigmoid(weights*inputs+bias) where * denotes the dot product"""
    for b,w in zip(self.biases, self.weights):
      a = sigmoid(np.dot(w,a)+b)
    return a

  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    """trains the neural network with stochastic gradient descent
    testing not implemented yet"""
    n=len(training_data)
    for j in range(epochs):
      random.shuffle(training_data)
      mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta)
      print('Epoch {0} complete'.format(j))
    


#General functions used
def sigmoid(z):
  """The sigmoid function."""
  return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
  """Derivative of the sigmoid function."""
  return sigmoid(z)*(1-sigmoid(z))

x=Network([2,3,2])
print(x)
print(x.feedforward(np.array([1,1])))