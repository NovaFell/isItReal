import numpy as np
import matplotlib.pyplot as plt


# Initialize the parameters of the neural network
# layer_dims: list containing the dimensions of each layer in the network
def init_params(layer_dims):
    np.random.seed(3)
    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return params

# Define the sigmoid function
# Computes the value of the sigmoid function for any given value Z
# Stores cache values for backpropagation
# Z (linear hypothesis) - Z = W*X + b
# W - weight matrix, b - bias vector, X - input vector
def sigmoid(Z):
    A = 1/(1+np.exp(np.dot(-1, Z)))
    cache = Z
    return A, cache


#Define forward propagation
# Loop over the layers of the neural network and compute the linear hypothesis and activation for each layer
# Take the value of Z and give it to the sigmoid function to get the activation value
# Return the final activation value and the caches for backpropagation
def forward_prop(X, params):
    A = X # input to first layer i.e. training data
    caches = []
    L = len(params)//2
    for l in range(1, L+1):
        A_prev = A

        # Linear Hypothesis
        Z = np.dot(params['W'+str(l)], A_prev) + params['b'+str(l)]

        # Storing the linear cache
        linear_cache = (A_prev, params['W'+str(l)], params['b'+str(l)])

        # Applying sigmoid on linear hypothesis
        A, activation_cache = sigmoid(Z)

        # Storing the both linear and activation cache
        cache = (linear_cache, activation_cache)
        caches.append(cache)

    return A, caches

# Define cost function
# As the value of the cost decreases, the performance of our model becomes better.
def cost_function(A,Y):
    m = Y.shape[1]

    cost = (-1/m)*(np.dot(np.log(A), Y.T) + np.dot(np.log(1-A), 1-Y.T))

    return cost


# Define the function for backpropagation for one step
# Calculate the gradient values for sigmoid units of one layer using the cache values
# Then calculate the derivates of the cost function with repect to weights, biases, and previous activation
def one_layer_backward(dA, cache):
    linear_cache, activation_cache = cache

    Z = activation_cache
    dZ = dA*sigmoid(Z)*(1-sigmoid(Z)) # The derivative of the sigmoid function

    A_prev, W, b = linear_cachem = A_prev.shape[1]
    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

# Define the backpropogation function
# Once we have looped through all the layers and computed the gradients, we store them in a grads dictionary and return it
def backprop(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    current_cache = caches[L-1]
    grads['dA'+str(L-1)], grads['dW'+str(L-1)], grads['db'+str(L-1)] = one_layer_backward(dAL, current_cache)

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = one_layer_backward(grads['dA'+str(l+1)], current_cache)
        grads['dA'+str(l)] = dA_prev_temp
        grads['dW'+str(l)] = dW_temp
        grads['db'+str(l)] = db_temp

    return grads

# Define the update parameters function
# Use the gradients calculated in backpropagation to update the weights and biases of the neural network
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2

    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*grads['W'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*grads[['b'+str(l+1)]]

    return parameters

# Define the train function, which puts everything defined above together
# This will go through all the functions step by step for a given number of epochs then return
# the final updated parameters and the cost history. 
# The cost history can be used to evaluate the performance of your network architecture.
def train(X, Y, layer_dims, epochs, lr):
    params = init_params(layer_dims)
    cost_history = []

    for i in range(epochs):
        Y_hat, caches = forward_prop(X, params)
        cost = cost_function(Y_hat, Y)
        cost_history.append(cost)
        grads = backprop(Y_hat, Y, caches)

        params = update_parameters(params, grads, lr)

    return params, cost_history