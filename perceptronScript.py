"""Run a neural network for digit recognition via stochastic gradient descent"""

# Load data
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Initialise neural network
import perceptron
net = perceptron.Network([784,30,10])

# train the network
net.SGD(training_data, 30,10,.1, test_data=test_data)