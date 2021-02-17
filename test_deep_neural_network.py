# -*- coding: utf-8 -*-
"""
Unit Test for deep_neural_network.py
Use the data from Coursera Neural Nteworks and Deep Learning Class week 4
to verify the functions are correct.

I am creating the same exercise outside of the class Jupyter Note Book to 
ensure I understand what is going on and don't miss something.
   
"""
# Import python modules
import unittest    #Unittest class
import numpy as np

# Import testcase init functions
from testCases_v4a import linear_forward_test_case
from testCases_v4a import linear_activation_forward_test_case
from testCases_v4a import L_model_forward_test_case_2hidden
from testCases_v4a import compute_cost_test_case
from testCases_v4a import linear_backward_test_case
from testCases_v4a import linear_activation_backward_test_case
from testCases_v4a import L_model_backward_test_case
from testCases_v4a import update_parameters_test_case
from dnn_app_utils_v3 import load_data


# Import functions to unittest
from deep_neural_network import initialize_params
from deep_neural_network import initialize_parameters_deep
from deep_neural_network import linear_forward
from deep_neural_network import linear_activation_forward
from deep_neural_network import L_model_forward
from deep_neural_network import compute_cost
from deep_neural_network import linear_backward
from deep_neural_network import linear_activation_backward
from deep_neural_network import L_model_backward
from deep_neural_network import update_params
from deep_neural_network import two_layer_model
from deep_neural_network import normalize_inputs
from deep_neural_network import L_layer_model


# Define test class that inherits from unittest.testcase() #########
class TestDeepNeuralNetwork(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(1) #Set a seed so results are consistant.
        
    def teardown(self):
        pass #Don't do any teardown
        
    def test_visualize_the_data(self):
         pass #Place keeper if want to plot input data       
#        plt.scatter(self.X[0, :], self.X[1, :], 
#                    c=self.Y, s=40, cmap=plt.cm.Spectral)   

    def test_initialize_params(self):
        """
        Unit Test for initialize_parameters
        
        Prototype:
            def initialize_parameters(n_x, n_h, n_y, scale = 0.01):
        """    
        
        #Arrange
        seed = 1
        n_x = 3
        n_h = 2
        n_y = 1
        expected_W1 = np.array([
                                [ 0.01624345, -0.00611756, -0.00528172],
                                [-0.01072969,  0.00865408, -0.02301539]
                               ])
        expected_b1 = np.array([
                       [0.],
                       [0.]])
        expected_W2 = np.array([
                       [0.01744812, -0.00761207]])
        expected_b2 = np.array([[0.]])
        
                      
        #Act
        params = initialize_params(n_x, n_h, n_y, scale = 0.01, seed = seed)
        
        _err_msg = ('\n\n'
                    f'Expected W2 =\n {expected_W2}\n\n'
                    f'Received W2 =\n {params["W2"]}\n\n'
                    
                    f'Expected b2 =\n {expected_b2}\n\n'
                    f'Received b2 =\n {params["b2"]}\n\n'
                    
                    f'Expected W1 =\n {expected_W1}\n\n'
                    f'Received W1 =\n {params["W1"]}\n\n'
                    
                    f'Expected b1 =\n {expected_b1}\n\n'
                    f'Received b1 =\n {params["b1"]}\n\n'
                   ) 
        

        #Assert
        np.testing.assert_array_almost_equal(expected_W2, 
                                             params["W2"],
                                             err_msg = _err_msg)
        
        np.testing.assert_array_almost_equal(expected_b2, 
                                             params["b2"], 
                                             err_msg = _err_msg)
        
        np.testing.assert_array_almost_equal(expected_W1, 
                                             params["W1"], 
                                             err_msg = _err_msg)
        
        np.testing.assert_array_almost_equal(expected_b1, 
                                             params["b1"], 
                                             err_msg = _err_msg)
        
    def test_initialize_parameters_deep(self):
        """
        Unit Test for initialize_parameters
        
        Prototype:
        Unit Test for initialize_parameters
        
        Prototype:
            def initialize_parameters_deep(layer_dims, scale = scale, seed = seed):
        """
        
        #Arrange
        seed = 3
        scale = 0.01
        layer_dims = [5, 4, 3]
        
        exp_W1 = np.array([
            [ 0.01788628,  0.0043651,   0.00096497, -0.01863493, -0.00277388],
            [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
            [-0.01313865,  0.00884622,  0.00881318,  0.01709573,  0.00050034],
            [-0.00404677, -0.0054536,  -0.01546477,  0.00982367, -0.01101068]
            ])
        exp_b1 = np.array([
                          [ 0.],
                          [ 0.],
                          [ 0.],
                          [ 0.]
                          ])
        exp_W2 = np.array([
                          [-0.01185047, -0.0020565,  0.01486148,  0.00236716],
                          [-0.01023785, -0.00712993, 0.00625245, -0.00160513],
                          [-0.00768836, -0.00230031, 0.00745056,  0.01976111]
                          ])
        exp_b2 = np.array([
                          [ 0.],
                          [ 0.],
                          [ 0.]
                          ])
        
        #Act
        params = initialize_parameters_deep(layer_dims, 
                                            scale = scale, 
                                            seed = seed)
        #Build error messages
        _err_msg = ""
        
        for l in range(1, len(layer_dims)):
            _err_msg = _err_msg + (f'Expected W{l} = {params["W" + str(l)]} \n'
                                   f'Expected b{l} = {params["b" + str(l)]} \n')
        
        #Assert
        np.testing.assert_array_almost_equal(exp_W2, 
                                             params["W2"],
                                             err_msg = _err_msg)
        
        np.testing.assert_array_almost_equal(exp_b2, 
                                             params["b2"], 
                                             err_msg = _err_msg)
        
        np.testing.assert_array_almost_equal(exp_W1, 
                                             params["W1"], 
                                             err_msg = _err_msg)
        
        np.testing.assert_array_almost_equal(exp_b1, 
                                             params["b1"], 
                                             err_msg = _err_msg)
        
    def test_linear_forward(self):                                             
        """
        Unit Test for linear_forward
        
        Prototype:
        def linear_forward(A, W, b):
        """
        
        #Arrange
        A, W, b = linear_forward_test_case()
        exp_Z = np.array([
                         [ 3.26295337, -1.23429987]
                         ])

        #Act
        Z, linear_cache = linear_forward(A, W, b)
        _err_msg = (f'Expected Z = {exp_Z} \n'
                    f'Received Z = {Z}')        

        #Assert                                              
        np.testing.assert_array_almost_equal(exp_Z, 
                                             Z, 
                                             err_msg = _err_msg)                                      
                                              
    def test_linear_activation_forward(self):                                             
        """
        Unit Test for linear_activation_forward
        
        Prototype:
        def linear_activation_forward(A_prev, W, b, activation):
        """
        #Arrange
        A_prev, W, b = linear_activation_forward_test_case()
        exp_A_sigmoid = np.array([[ 0.96890023, 0.11013289]])
        exp_A_relu = np.array([[ 3.43896131, 0.]])
        
        #Act
        #Use 'l_a_...' as abbreviation for 'linear_activation_...'
        A_sigmoid, l_a_cache = linear_activation_forward(A_prev, W, b, "sigmoid")
        A_relu, l_a_cache = linear_activation_forward(A_prev, W, b, "relu")
        _err_msg_sigmoid = (f'Expected A_sigmoid = {exp_A_sigmoid}\n'
                            f'Received A_sigmoid = {A_sigmoid}')        
        _err_msg_relu = (f'Expected A_relu = {exp_A_relu}\n'
                         f'Received A_relu = {A_relu}')
        
        #Assert
        np.testing.assert_array_almost_equal(exp_A_sigmoid, 
                                             A_sigmoid, 
                                             err_msg = _err_msg_sigmoid)  
        np.testing.assert_array_almost_equal(exp_A_relu, 
                                             A_relu, 
                                             err_msg = _err_msg_relu)           
        
      
    def test_L_model_forward(self):  
        """ Unit Test for L_model_forward  
        
        Prototype: def L_model_forward(X, parameters):
        """
        
        #Arrange
        X, params = L_model_forward_test_case_2hidden()
        exp_AL = np.array([[ 0.03921668, 0.70498921, 0.19734387, 0.04728177]])
        exp_cashes_len = 3               

        #Act
        AL, forward_caches = L_model_forward(X, params)
        AL_err_msg = (f'Expected AL = {exp_AL}\n'
                    f'Received AL = {AL}')
        cashes_len_err_msg = (f'Expected forward_caches len = '
                              f'{exp_cashes_len}\n'
                              f'Received forward_caches len = '
                              f'{len(forward_caches)}')
        
        #Assert
        np.testing.assert_array_almost_equal(exp_AL, AL, err_msg = AL_err_msg)
        self.assertEqual(exp_cashes_len, 
                         len(forward_caches), 
                         cashes_len_err_msg)
        
    def test_compute_cost(self):
        """ Unit Test for compute_cost
        
        Prototupe: def compute_cost(AL, Y):
        """
        
        #Arrange
        Y, AL = compute_cost_test_case()
        exp_cost = 0.2797765635793422
        
        #Act
        cost = compute_cost(AL, Y)
        _err_msg = (f'Expected cost = {exp_cost}\n'
                    f'Received cost = {cost}')
        
        #Assert
        self.assertEqual(exp_cost, cost, _err_msg)
        
    def test_linear_backward(self):
         """ Unit Test for linear_backward
         
         Prototype: def linear_backward(dZ, cache):
         """
         
         #Arrange
         dZ, linear_cache = linear_backward_test_case()
         
         exp_dA_prev = np.array([
                      [-1.15171336,  0.06718465, -0.3204696,   2.09812712],
                      [ 0.60345879, -3.72508701,  5.81700741, -3.84326836],
                      [-0.4319552,  -1.30987417,  1.72354705,  0.05070578],
                      [-0.38981415,  0.60811244, -1.25938424,  1.47191593],
                      [-2.52214926,  2.67882552, -0.67947465,  1.48119548]])
         
         exp_dW = np.array([
             [ 0.07313866, -0.0976715,  -0.87585828,  0.73763362,  0.00785716],
             [ 0.85508818,  0.37530413, -0.59912655,  0.71278189, -0.58931808],
             [ 0.97913304, -0.24376494, -0.08839671,  0.55151192,-0.10290907]])
         
         exp_db = np.array ([
                  [-0.14713786],
                  [-0.11313155],
                  [-0.13209101]])
         
         #Act
         dA_prev, dW, db = linear_backward(dZ, linear_cache)
         _err_msg_dA_prev = (f'Expected dA_prev =\n {exp_dA_prev}\n\n'
                             f'Received dA_prev =\n {dA_prev}\n')
         _err_msg_dW = (f'Expected dW =\n {exp_dW}\n\n'
                             f'Received dW =\n {dW}\n')
         _err_msg_db= (f'Expected db =\n {exp_db}\n\n'
                       f'Received db =\n {db}\n')    
         
         #Assert
         np.testing.assert_array_almost_equal(exp_dA_prev, 
                                             dA_prev, 
                                             err_msg = _err_msg_dA_prev)
         
         np.testing.assert_array_almost_equal(exp_dW, 
                                             dW, 
                                             err_msg = _err_msg_dW)    
         
         np.testing.assert_array_almost_equal(exp_db, 
                                             db, 
                                             err_msg = _err_msg_db)     
         
    def test_linear_activation_backward(self):
        """ Unit Test for linear_activation_backward
        
        Prototype: def linear_activation_backward(dA, cache, activation):
        """
        
        #Arrange
        dAL, linear_activation_cache = linear_activation_backward_test_case()
        
        exp_sig_dA_prev = np.array([
                                  [ 0.11017994,  0.01105339],
                                  [ 0.09466817,  0.00949723],
                                  [-0.05743092, -0.00576154]])
        exp_sig_dW = np.array([[ 0.10266786,  0.09778551, -0.01968084]])
        exp_sig_db = np.array([[-0.05729622]])
        
        exp_relu_dA_prev = np.array([[ 0.44090989, -0.        ],
                                     [ 0.37883606, -0.        ],
                                     [-0.2298228,   0.        ]])
        exp_relu_dW = np.array([[ 0.44513824,  0.37371418, -0.10478989]])
        exp_relu_db = np.array([[-0.20837892]])
        
        #Act
        sig_dA_prev, sig_dW, sig_db = linear_activation_backward(dAL, 
                                                    linear_activation_cache, 
                                                    activation = "sigmoid")
        relu_dA_prev, relu_dW, relu_db = linear_activation_backward(dAL, 
                                                    linear_activation_cache, 
                                                    activation = "relu")
        
        _err_msg_sig = (
                        f'Expected sig_dA_prev=\n'
                        f'{exp_sig_dA_prev}\n'
                        f'Received sig_dA_prev =\n'
                        f'{sig_dA_prev}\n\n'
        
                        f'Expected sig_dW=\n'
                        f'{exp_sig_dW}\n'
                        f'Received sig_dW =\n'
                        f'{sig_dW}\n\n'
        
                        f'Expected sig_db =\n'
                        f'{exp_sig_db}\n'
                        f'Received sig_db =\n'
                        f'{sig_db}\n\n')
                        
        _err_msg_relu = (
                        f'Expected relu_dA_prev=\n'
                        f'{relu_dA_prev}\n'
                        f'Received relu_dA_prev =\n'
                        f'{relu_dA_prev}\n\n'                        
                        
        
                        f'Expected relu_dW =\n'
                        f'{exp_relu_dW}\n'
                        f'Received relu_dW =\n'
                        f'{relu_dW}\n\n'                        
                        
        
                        f'Expected relu_db =\n'
                        f'{exp_relu_db}\n'
                        f'Received relu_db =\n'
                        f'{relu_db}\n\n'                        
                        )
        
        #Assert
        np.testing.assert_array_almost_equal(exp_sig_dA_prev, 
                                             sig_dA_prev, 
                                             err_msg = _err_msg_sig)    
        np.testing.assert_array_almost_equal(exp_sig_dW, 
                                             sig_dW, 
                                             err_msg = _err_msg_sig)   
        np.testing.assert_array_almost_equal(exp_sig_db, 
                                             sig_db, 
                                             err_msg = _err_msg_sig)   
        
        np.testing.assert_array_almost_equal(exp_relu_dA_prev, 
                                             relu_dA_prev, 
                                             err_msg = _err_msg_relu)    
        np.testing.assert_array_almost_equal(exp_relu_dW, 
                                             relu_dW, 
                                             err_msg = _err_msg_relu)   
        np.testing.assert_array_almost_equal(exp_relu_db, 
                                             relu_db, 
                                             err_msg = _err_msg_relu)   
    def test_L_model_backward(self):
        """ Unit test for L_model_backward
        
        prototype: def L_model_backward(AL, Y, caches):
        """
        #Arrange
        AL, Y_assess, caches = L_model_backward_test_case()
        exp_dW1 = np.array([
            [ 0.41010002,  0.07807203,  0.13798444,  0.10502167],
            [ 0.,          0.,          0.,          0.        ],
            [ 0.05283652,  0.01005865,  0.01777766,  0.0135308 ]])
        exp_db1 = np.array([
                            [-0.22007063],
                            [ 0.        ],
                            [-0.02835349]])
        exp_dA1 = np.array([
                           [ 0.12913162, -0.44014127],
                           [-0.14175655,  0.48317296],
                           [ 0.01663708, -0.05670698]])
        
        #Act
        grads = L_model_backward(AL, Y_assess, caches)
        _err_msg = (f'Expected dW1 = \n {exp_dW1}/n/n'
                    f'Received dW1 = \n {grads["dW1"]}/n/n'
                    
                    f'Expected dW1 = \n {exp_db1}/n/n'
                    f'Received dW1 = \n {grads["db1"]}/n/n'       
                    
                    'Expected dA1 = \n {exp_dA1}/n/n'
                    f'Received dA1 = \n {grads["dA1"]}/n/n'                      
                    )
        
        #Assert
        np.testing.assert_array_almost_equal(exp_dW1, 
                                             grads["dW1"], 
                                             err_msg = _err_msg)    
        np.testing.assert_array_almost_equal(exp_db1, 
                                             grads["db1"], 
                                             err_msg = _err_msg)   
        np.testing.assert_array_almost_equal(exp_dA1, 
                                             grads["dA1"], 
                                             err_msg = _err_msg)   
        
    def test_update_params(self):
        """ Unittest for update_params
        
        Prototype: def update_parameters(parameters, grads, learning_rate):
        """
        #Arrange
        params, grads = update_parameters_test_case()
        exp_W1 = np.array([
            [-0.59562069, -0.0999178,  -2.14584584,  1.82662008],
            [-1.76569676, -0.80627147,  0.51115557, -1.18258802],
            [-1.0535704,  -0.86128581,  0.68284052,  2.20374577]])
        exp_b1 = np.array([[-0.04659241],
                           [-1.28888275],
                           [ 0.53405496]])
        exp_W2 = np.array([[-0.55569196,  0.0354055,   1.32964895]])
        exp_b2 = np.array([[-0.84610769]])
        
        #Act
        params = update_params(params, grads, 0.1)
        _err_msg = (f'Expected W1 = \n{exp_W1}\n\n'
                    f'Received W1 = \n{params["W1"]}\n\n'
                    
                    f'Expected b1 = \n{exp_b1}\n\n'
                    f'Received b1 = \n{params["b1"]}\n\n'
                    
                    f'Expected W2 = \n{exp_W2}\n\n'
                    f'Received W2 = \n{params["W2"]}\n\n'
                    
                    f'Expected b2 = \n{exp_b2}\n\n'
                    f'Received b2 = \n{params["b2"]}\n\n'
                    )
        
        #Assert
        np.testing.assert_array_almost_equal(exp_W1, 
                                             params["W1"], 
                                             err_msg = _err_msg)    
        np.testing.assert_array_almost_equal(exp_b1, 
                                             params["b1"], 
                                             err_msg = _err_msg)   
        
        np.testing.assert_array_almost_equal(exp_W2, 
                                             params["W2"], 
                                             err_msg = _err_msg)    
        np.testing.assert_array_almost_equal(exp_b2, 
                                             params["b2"], 
                                             err_msg = _err_msg)  
        

class TestDeepNeuralNetworkClassification(unittest.TestCase):
    
    def setUp(self):
        """ setup unit test for DeepNeuralNetworkClassifcation
        
            The test data is loaded using load_data() and normalized using 
            normaize_inputs
        """
        np.random.seed(1) #Set a seed so results are consistant.
        # Get the test data
        train_x_orig, self.train_y, test_x_orig, self.test_y, classes = load_data()
        # flatten and normalize the data
        self.train_X, self.test_X = normalize_inputs(train_x_orig, test_x_orig)
        # determan input and output model dims
        # hidden layer dimensions define in test cases since they are not 
        # the same for all models
        self.n_x = self.train_X.shape[0]        
        self.n_y = self.train_y.shape[0]      
        
        
    def teardown(self):
        pass #Don't do any teardown
        
    def test_two_layer_model(self):
        """
        Unit test for two_layer_model
        prototype: def two_layer_model(X, Y, layers_dims, 
                                        learning_rate = 0.0075,
                                        num_iterations = 2500):
        """
        # Arrange
        n_h = 7
        layers_dims = (self.n_x, n_h, self.n_y)
        test_cases = {
                      0 : 0.6930497356599888,
                      100 : 0.6464320953428849,
                      2400 : 0.048554785628770115
                      }

        
        # Act
        print('Running 2 Level Model\n')
        params, costs = two_layer_model(self.train_X, self.train_y, 
                                        layers_dims,  num_iterations = 2500,
                                        print_costs = True,
                                        seed = 1)
        
        
        # Assert
        for test_vector in test_cases.keys():
            _err_msg = (f'\n\nExpected costs[{test_vector}] = '
                        f'{test_cases[test_vector]}\n'
                        f'Received costs[{test_vector}] = '
                        f'{costs[test_vector]}')
            self.assertAlmostEqual(costs[test_vector], 
                              test_cases[test_vector],
                              places = 6,
                              msg = _err_msg)
        
        
    def test_L_layer_model(self):
        """
        Unit test for two_layer_model
        prototype: def L_layer_model(X, Y, layers_dims, 
                                     learning_rate = 0.0075, 
                                     num_iterations = 3000, 
                                     print_cost=False, 
                                     seed = None):
       """
        # Arrange
        layers_dims = [self.n_x, 20, 7, 5,  self.n_y] #  4-layer model

        test_cases = {
                      0 : 0.771749,
                      100 : 0.672053,
                      2400 : 0.092878
                     }

        
        # Act
        print('Running L Level Model\n')
        params, costs = L_layer_model(self.train_X, self.train_y, 
                                      layers_dims,
                                      learning_rate = 0.0075,
                                      num_iterations = 2500,
                                      print_costs = True,
                                      seed = 1)
        
        
        # Assert
        for test_vector in test_cases.keys():
            _err_msg = (f'\n\nExpected costs[{test_vector}] = '
                        f'{test_cases[test_vector]}\n'
                        f'Received costs[{test_vector}] = '
                        f'{costs[test_vector]}')
            self.assertAlmostEqual(costs[test_vector], 
                             test_cases[test_vector],
                             places = 6,
                             msg = _err_msg)
        
     

if __name__ == '__main__':
    unittest.main()

         