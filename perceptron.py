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

  def SGD(self, training_data, repeats, mini_batch_size, rate, test_data=None):
    """trains the neural network with stochastic gradient descent
    testing not implemented yet"""
    if test_data: n_test=len(test_data)
    n=len(training_data)
    for j in range(repeats):
      random.shuffle(training_data)
      mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, rate)
      if test_data:
        print('Repeat {0}: {1} / {2}'.format(j+1, self.evaluate(test_data), n_test))
      else:
        print('Repeat {0} complete'.format(j))
    
  def update_mini_batch(self, mini_batch, rate):
    """update weights and biases by applying gradient descent using backpropagation to a single mini batch of input output tuples"""
    # initialize changes to biases and weights
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # calculate contribution to backpropagation from each input and sum
    for x, y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x,y)
      nabla_b=[nb+d for nb, d in zip(nabla_b, delta_nabla_b)]
      nabla_w=[nw+d for nw, d in zip(nabla_w,delta_nabla_w)]
    self.weights = [w-(rate/len(mini_batch))*nw for w, nw in zip(self.weights,nabla_w)]
    self.biases = [b-(rate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

  def backprop(self, x, y):
    """"Return a tuple ''(nabla_b,nabla_w)'' representing the gradient of the cost function"""
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # feedforward
    activation = x
    activations = [x]
    zs= []
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, activation)+b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)
    # single backward pass
    delta = (activations[-1]-y)*sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    # loop for rest of backwards passes
    for l in range(2, self.num_layers):
      z = zs[-l]
      sp = sigmoid_prime(z)
      delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) 
    return (nabla_b,nabla_w)

  def evaluate(self, test_data):
    test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
    return sum(int(x==y) for (x,y) in test_results)

#General functions used
def sigmoid(z):
  """The sigmoid function."""
  return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
  """Derivative of the sigmoid function."""
  return sigmoid(z)*(1-sigmoid(z))