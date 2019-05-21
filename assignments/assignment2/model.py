import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        hidden_layer_size, int - number of neurons in the hidden layer
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        reg, float - L2 regularization strength
        """
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_layer_size = hidden_layer_size
        self.reg = reg
        # TODO Create necessary layers
        self.first_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.ReLu_layer = ReLULayer()
        self.second_layer = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        first_layer_params  = self.first_layer.params()
        first_layer_params['W'].grad  = np.zeros_like(first_layer_params['W'].grad)
        first_layer_params['B'].grad  = np.zeros_like(first_layer_params['B'].grad)

        second_layer_params = self.second_layer.params()
        second_layer_params['W'].grad = np.zeros_like(second_layer_params['W'].grad)
        second_layer_params['B'].grad = np.zeros_like(second_layer_params['B'].grad)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        first_l_output  = self.first_layer.forward(X)
        relu_output   = self.ReLu_layer.forward(first_l_output)
        second_l_output = self.second_layer.forward(relu_output)

        loss_reg_first, grad_reg_first = l2_regularization(first_layer_params['W'].value, self.reg)
        loss_reg_second, grad_reg_second = l2_regularization(second_layer_params['W'].value, self.reg)

        loss, gradient = softmax_with_cross_entropy(second_l_output, y)
        loss = loss + loss_reg_first + loss_reg_second

        gradient_by_second = self.second_layer.backward(gradient)
        second_layer_params['W'].grad += grad_reg_second

        gradient_by_ReLu = self.ReLu_layer.backward(gradient_by_second)

        gradient_by_first = self.first_layer.backward(gradient_by_ReLu) 
        first_layer_params['W'].grad += grad_reg_first

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        first_l_output = self.first_layer.forward(X)
        relu_output = self.ReLu_layer.forward(first_l_output)
        second_l_output = self.second_layer.forward(relu_output)

        pred = np.argmax(second_l_output, axis=1)
    
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        first_layer_params = self.first_layer.params()
        result['W1'] = first_layer_params['W']
        result['B1'] = first_layer_params['B']

        second_layer_params = self.second_layer.params()
        result['W2'] = second_layer_params['W']
        result['B2'] = second_layer_params['B']

        return result
