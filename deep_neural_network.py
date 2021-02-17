# -*- coding: utf-8 -*-
"""
Implement the functions required to build a deep neural network.

After this assignment I hope to be able to:
    Use non-linear units like ReLU
    Build a neural network with more than 1 hidden layer
    Implement a neural network class
    
    Note the Notebook uses np.random.seed(1) for the homework.  Need to use
    this when writing unittests if using HW data.

"""

#Imports
import numpy as np

from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

#Module Level Variables
#Put any module variables here
None

def normalize_inputs(train_X_orig, test_X_orig, norm_factor = 1/255):
    """
    Flatten and normailze the training and test inputs.
    
    Behavior
    --------
    Takes a (m, x_pix, y_pix, 3) input np.ndarray and flattens it into a 
    2 dimensional (x_pix*y_pix*3, m) np.ndarray.
    
    It then normalizes it by multiplying each value by the norm_factor
    
    Parameters
    ----------
    train_X_orig : np.ndarray training inputs
                   (num_train_examples, x_pix, y_pix, depth)
                   num_train_examples : int, m_train
                   x_pix : int, number of pixels in the x direction
                   y_pix : int, number of pixels in the y direction
                   depth : int, color depth, 3 for RGB
    test_X_orig : np.ndarray test inputs
                   (num_test_samples, x_pix, y_pix, depth)
                   num_test_samples : int, m_test
                   x_pix : int, number of pixels in the x direction
                   y_pix : int, number of pixels in the y direction
                   depth : int, color depth, 3 for RGB
    

    Returns
    -------
        train_X, test_X : tuple of normalized and flattened training and 
        test input np.ndarrays
            train_X : np.ndarray
                  (x_pix * y_pix * depty x num_train_examples)
            test_X : np.ndarray
                  (x_pix * y_pix * depty x num_test_samples)
    """
    # Useful Variables
    None
    
    # Reshape the training and test examples 
    # The "-1" makes reshape flatten the remaining dimensions
    train_X_flatten = train_X_orig.reshape(train_X_orig.shape[0], -1).T  
    test_X_flatten = test_X_orig.reshape(test_X_orig.shape[0], -1).T
    
    # Normalize the inputs across the last dimension 
    train_X = train_X_flatten * norm_factor    
    test_X = test_X_flatten * norm_factor

    return (train_X, test_X)
    
        

def initialize_params(n_x, n_h, n_y, scale = 0.01, seed = None):
    """

    Parameters
    ----------
    n_x : int
         n_x = n[0] = number of inputs in the model.
    n_h : int
        n_h = n[1] = number of hidden nodes in model
    n_y : int
        n_y = n[2] = number of outputs in the model
    scale : float, optional
        factor to scale the initialization parameters by to center around 0
    seed : int, optional
        Set seed value to cause the initialization to be consistant.  IOW, the 
        initialization will generate the same random numbers to be placed in 
        the parameters.  This will allow for unittesting to check the results.
        The default is None, so complete randomization will occur.
        

    Returns
    -------
    params -- python dictionary containing your parameters:
        "W1":W1 -- randomized weight matrix of 1st layer (n_h, n_x) or n[1] x n[0]
        "b1":b1 -- randomized bias vector of 1st layer (n_h, 1) or n[1] x 1
        "W2":W2 -- randomized weight matrix of 2nd layer (n_y, n_h) n[2] x n[1]
        "b2":b2 -- randomized bias vector of 2nd layer (n_y, 1) or n[2] x 1
    """  
    # if seed passed in set seed to produce consistant random numbers to 
    # allow for unittesting
    if seed: np.random.seed(seed)
    
    params = {}
    params["W1"] = np.random.randn(n_h, n_x) * scale
    params["b1"] = np.zeros((n_h, 1))
    params["W2"] = np.random.randn(n_y, n_h) * scale
    params["b2"] = np.zeros((n_y, 1))

    return params

def initialize_parameters_deep(layer_dims, scale = 0.01, seed = None):
    """
    Initialization for an L-layer Neural Network.  Weights W[l] are initialize
    randomly, and b[l] are set to zero

    Parameters
    ----------
    layer_dims : list
               python array containing the dimensions of each layer in our 
               network.  Not np.array
    scale : float, optional
        factor to scale the initialization parameters by to center around 0.
        Dr. Ng used scale = 1 / np.sqrt(layer_dims[l-1]) DO NOT KNOW WHY
        if scale == 'sqrt' then scale = 1 / np.sqrt(layer_dims[l-1])
        The default is 0.01
    seed : int, optional
               used to provide a consistant randomization to enable unittest.
               The default is None
    Returns
    -------
    params -- python dictionary containing your parameters "W1", "b1", 
              ..., "WL", "bL":
                  Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                  bl -- bias vector of shape (layer_dims[l], 1)

    """
    # if seed passed in set seed to produce consistant random numbers to 
    # allow for unittesting
    if seed: np.random.seed(seed)

    #Useful varibles
    params = {}
    L = len(layer_dims)

    

    
    #loop through layers
    for l in range(1, L):
        # Build keys to make code easier to read
        W = 'W' + str(l)
        b = 'b' + str(l)
        
        #calculate scale... see comment in doc string
        if scale == 'sqrt':
            _scale = 1 / np.sqrt(layer_dims[l-1])
        else:
            _scale = scale
        
        #Randomize and scale Weights, set offsets to zero
        params[W] = np.random.randn(layer_dims[l],
                                    layer_dims[l-1]) * _scale
        params[b] = np.zeros((layer_dims[l], 1))
        
    #Sanaty check dimensions
    assert(params[W].shape == (layer_dims[l], layer_dims[l-1]))
    assert(params[b].shape == (layer_dims[l], 1))
        
    return params

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Parameters:
    A : np.ndarray
        Activations from previous layer (or input data)
        (size of previous layer, number of examples)
    W : np.ndarray
        weights matrix
        (size of current layer, size of previous layer)
    b : np.ndarray
        bias vector
        (size of the current layer, 1)
  
    Returns:
    Z, cache : np.ndarray, tuple 
        Z : np.ndarray
           The input of the activation function, 
           also called pre-activation parameter 
       cache : tuple
           a python tuple containing "A", "W" and "b"
           stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache   
    
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev  : np.ndarray
        Activations from previous layer (or input data)
        (size of previous layer, number of examples)
    W : np.ndarray
        weights matrix
        (size of current layer, size of previous layer)
    b : np.ndarray
        bias vector
        (size of the current layer, 1)-- activations from previous layer (or input data): (size of previous layer, number of examples)
    activation : string
                 The activation to be used in this layer, 
                 stored as a text string: "sigmoid" or "relu"

    Returns:
    A, cache : np.ndarray, tuple
        A : np.ndarray
            The output of the activation function, 
            also called the post-activation value 
    cache : linear_cache, activation_cache
             stored for computing the backward pass efficiently
            linear_cache : tuple
                           a python tuple containing "A", "W" and "b"
                           stored for computing the backward pass efficiently
            activation_cache: np.ndarray
                              Z
                              (size of current layer, number of examples)                                       
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, params):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X : np.ndarray
        Input Data
        (input size, number of examples)
    params: dict
            output of initialize_parameters_deep()
    
    Returns:
    AL , forward_caches : np.ndarray, list
        AL: np.ndarray
            last post-activation value (y_hat)
    forward_caches : list
             caches containing:
                every cache of linear_activation_forward()
                there are L of them, indexed from 0 to L-1
                caches[0] is the input X
    """
    #Useful variables
    forward_caches = []
    A = X
    L = len(params) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        #Point at next layer
        A_prev = A 
        W = params['W' + str(l)]
        b = params['b' + str(l)]
        
        #Calculate next layer and save
        A, cache = linear_activation_forward(A_prev, W, b, 'relu')
        forward_caches.append(cache)

    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.

    A_prev = A
    W = params['W' + str(L)]
    b = params['b' + str(L)]    
    AL, cache = linear_activation_forward(A_prev, W, b, 'sigmoid')
    forward_caches.append(cache)
    
    # Sanity Check of shapes
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, forward_caches

def compute_cost(AL, Y):
    """
    Implement the cost function.

    Arguments:
        AL : np.ndarray
         sigmoid output of the deep neural network forward propigation.
         (1 x m)
        Y : np.ndarray
         "true" labels vector of data 
         (1 x m)

    Returns:
        j : float
            cross-entropy cost given equation (13) in:
            https://www.coursera.org/learn/neural-networks-deep-learning/ungradedLab/l8YCO/lab
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    j = -1/m * ( np.dot(Y, np.log(AL).T) + np.dot(1-Y, np.log((1-AL).T)))     
    j = np.squeeze(j)       # To make sure your cost's shape is what we expect
                            # (e.g. this turns [[17]] into 17)
    #Sanatiy check shape of cost
    assert(j.shape == ())
    
    return j


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ : np.ndarray
         Gradient of the cost with respect to the linear output 
         (of current layer l)
    cache : tuple of values (A_prev, W, b) 
            coming from the forward propagation in the current layer
            e.g. forward_caches[l]

    Returns:
    dA_prev : np.ndarray
              Gradient of the cost with respect to the activation 
              (of the previous layer l-1), same shape as A_prev
    dW : np.ndarray
         Gradient of the cost with respect to W (current layer l), 
         same shape as W
    db : np.ndarray
         Gradient of the cost with respect to b (current layer l),
         same shape as b
    """
    #Define useful variables
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    #Sanity Check Answers using shapes of matricies
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA : np.ndarray 
         post-activation gradient for current layer l 
         this was calculated by running linear_activation_backward on layer l+1
    
    cache : tuple of linear_cache, activation_cache
            stored for computing the backward pass efficiently
            linear_cache : tuple
                           a python tuple containing A[l], W[l] and b[l]
                           stored during forward propigation for computing 
                           the backward pass efficiently
            activation_cache: np.ndarray
                              Z[l] used to calculate A[l]
                              (size of current layer, number of examples) 
    
    activation : string
                 the activation to be used in this layer, 
                 stored as a text string: "sigmoid" or "relu"    
    Returns:
    dA_prev : np.ndarray
              Gradient of the cost with respect to the activation 
              (of the previous layer l-1), same shape as A_prev.
              Note: the previous layer (l-1) is the next layer to be 
                    calculated since we are going backward.
    dW : np.ndarray
        Gradient of the cost with respect to W 
        (current layer l), same shape as W
    db : np.ndarray vector
         Gradient of the cost with respect to b 
         (current layer l), same shape as b
    """
    
    #define some useful variables
    linear_cache, activation_cache = cache
    
    # Calculate Gradients 
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL : np.ndarray
         probability vector, output of the forward propagation 
         L_model_forward()
         size (1 x m)
    Y : np.ndarray
        true "label" vector (containing 0 if False, 1 if True)
        size (m x 1)
    caches : list of caches[0 : l-1]
             cache[l-1] is from linear_actibation_forward with "sigmoid"
             cache[0 : l-2] is from linear_activation_forward with "relu"
             cache[] : tuple of linear_cache, activation_cache
                       linear_cache : tuple
                           a python tuple containing A[l], W[l] and b[l]
                           stored during forward propigation for computing 
                           the backward pass efficiently
                       activation_cache: np.ndarray
                           Z[l] used to calculate A[l]
                           (size of current layer, number of examples) 
                
    Returns:
    grads :  A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    #Useful Variables
    grads = {}
    L = len(caches) # the number of layers
    # m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation using the partial deravitive of 
    # cost with respect to AL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) 
    
    # Lth layer (SIGMOID -> LINEAR) gradients. 
    # Inputs: "dAL, current_cache". 
    # Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    _grads = linear_activation_backward(dAL, 
                                        current_cache, 
                                        activation = 'sigmoid')
    
    (grads["dA" + str(L-1)], 
     grads["dW" + str(L)], 
     grads["db" + str(L)]) = _grads
    
    # L-1 to 0th layer (relu -> LINEAR) gradients. 
    # Inputs: "grads["dA" + str(l + 1)], current_cache". 
    # Outputs: "grads["dA" + str(l)], 
    #           grads["dW" + str(l + 1)], 
    #           grads["db" + str(l + 1)] 
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        _grads = linear_activation_backward(grads["dA" + str(l+1)], 
                                            current_cache,
                                            activation = "relu")
        (grads["dA" + str(l)],
         grads["dW" + str(l+1)], 
         grads["db" + str(l+1)]) = _grads

    return grads



def update_params(params, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    params : python dictionary containing your parameters 
    
    params : python dictionary containing your parameters 
             "W1", "b1", ..., "WL", "bL":
                  Wl : np.ndarray
                       weight matrix of layer l 
                       (layer_dims[l] x layer_dims[l-1])
                  bl : np.ndarray
                       bias vector of layer l 
                       (layer_dims[l] x 1)
    
    grads : python dictionary containing your gradients,
            "dW1", "db1", ..., "dWL", "dbL":
                  dWl : np.ndarray
                       gradient for weight matrix of layer l 
                       (layer_dims[l] x layer_dims[l-1])
                 dbl : np.ndarray
                       gradient for bias vector of layer l 
                       (layer_dims[l] x 1)      
            output of L_model_backward
    
    Returns:
    params :  python dictionary containing your updated parameters 
              same definition of params input
    """
    
    L = len(params) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        params["W" + str(l+1)] = params["W" + str(l+1)] - grads["dW" + str(l+1)] * learning_rate
        params["b" + str(l+1)] = params["b" + str(l+1)] - grads["db" + str(l+1)] * learning_rate
    return params


def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, 
                    num_iterations = 3000, print_costs = False, seed = None):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Parameters:
    X : np.ndarray, input data
        (n_x x number of examples)
    Y : np.ndarray, true "label" vector 
        containing 1 if cat, 0 if non-cat
        (1 x number of examples)
    layers_dims : list, 
                  dimensions of the layers 
                  (n_x, n_h, n_y) for a 2 layer model
                  n_x : int, number of inputs
                  n_h : int, number of hidden layers
                  n_y : int, number of outputs
    num_iterations : int, optional
                     number of iterations of the optimization loop
                     defaults to 3000
    learning_rate : float, optional 
                    learning rate of the gradient descent update rule (alpha)
                    default to 0.0075 
    print_costs : bool, optional
                  if true print every 100th cost to the screen to show if
                  learning.
    
    Returns:
    tuple of params and cost
        params : dict of the prediction weights and offsets
                 params["W1"] weight for layer 1
                 params["b1"] offset for layer 1
                 params["W2"] weight for layer 2
                 params["b1"] offset for layer 2
        costs : dict of the costs for every ith iteration
                costs[i] cost of ith iteration.  Recorded every 100 iterations.
    """
    #Usefull variables
    grads = {}
    costs = {}                                 # to keep track of the cost
    # m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary
    params = initialize_params(n_x, n_h, n_y, seed = seed)
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation loop: 
        # LINEAR -> RELU -> LINEAR -> SIGMOID. 
        # Inputs: "X, W1, b1, W2, b2". 
        # Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X, W1, b1, activation = "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation = "sigmoid")
        
       
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        ### START CODE HERE ### (≈ 2 lines of code)
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 
                                                   activation = 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 
                                                   activation = 'relu')
        ### END CODE HERE ###
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        params = update_params(params, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = params["W1"]
        b1 = params["b1"]
        W2 = params["W2"]
        b2 = params["b2"]
        
        # Print the cost every 100 training example
        if i % 100 == 0:
            costs[i] = compute_cost(A2, Y)
            if print_costs:
                print(f'costs[{i}] = {costs[i]}')

    return params, costs

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, 
                  num_iterations = 3000, print_cost=False, 
                  print_costs = False, seed = None):
    
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Parameters:
    X : np.ndarray, input data
        (n_x x number of examples)
    Y : np.ndarray, true "label" vector 
        containing 1 if true, 0 if false
        (1 x number of examples)
    layers_dims : list, 
                  dimensions of the layers 
                  (n_x, l[1], l[2], ... l[L-1], n_y) 
                  n_x : int, number of inputs
                  l[i] : int, number of nodes in hidden l[i] 
                  n_y : int, number of outputs
    num_iterations : int, optional
                     number of iterations of the optimization loop
                     defaults to 3000
    learning_rate : float, optional 
                    learning rate of the gradient descent update rule (alpha)
                    default to 0.0075 
    print_costs : bool, optional
                  if true print every 100th cost to the screen to show if
                  learning.
    seed : int, optional
           used to provide a consistant randomization to enable unittest.
           The default is None
    
    Returns:
    tuple of params and cost
        params : dict of the prediction weights and offsets
                 params["W1"] weight for layer 1
                 params["b1"] offset for layer 1
                 params["W2"] weight for layer 2
                 params["b1"] offset for layer 2
                 params["Wl-2"] weignt for layer l-2
                 params[:bl-2] offset for layer l-2
        costs : dict of the costs for every ith iteration
                costs[i] cost of ith iteration.  Recorded every 100 iterations.
    """
    #Useful variables
    costs = {}                        # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    #params =  initialize_parameters_deep(layers_dims, seed = seed)
    params =  initialize_parameters_deep(layers_dims, 
                                         scale = 'sqrt',
                                         seed = 1)
    
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, params)
        
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters
        params = update_params(params, grads, learning_rate)
                
        # Print the cost every 100 training example
        if i % 100 == 0:
            costs[i] = compute_cost(AL, Y)
            if print_costs:
                print(f'costs[{i}] = {costs[i]}\n')
               
    return params, costs


#Set up so can run from command line        
if __name__ == "__main__":
    #Some simple testcases
    #Arrange
    print('No test cases in deep_neural_network.py, run unittest cases\n'
          'in test_deep_neural_network.py')
    #Act
    
    #Assert
