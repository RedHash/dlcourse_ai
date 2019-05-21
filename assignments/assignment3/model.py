import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers

        image_width, image_height, n_channels = input_shape
        self.conv1 = ConvolutionalLayer(in_channels=n_channels, out_channels=conv1_channels, filter_size=3, padding=1)
        self.relu2 = ReLULayer()
        self.maxpool3 = MaxPoolingLayer(4, 4)
        self.conv4 = ConvolutionalLayer(in_channels=conv1_channels, out_channels=conv2_channels, filter_size=3, padding=1)
        self.relu5 = ReLULayer()
        self.maxpool6 = MaxPoolingLayer(4, 4)
        self.flatten7 = Flattener()
        n_input_classes = int(conv2_channels * image_height * image_height / 4 / 4 / 4 / 4)
        self.fullyc8 = FullyConnectedLayer(n_input_classes, n_output_classes)

        self.reg = 1e-7

    def clear_gradients(self, params):
        params['W'].grad  = np.zeros_like(params['W'].grad)
        params['B'].grad  = np.zeros_like(params['B'].grad)   
        
        pass     

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        
        self.clear_gradients(self.conv1.params())
        self.clear_gradients(self.conv4.params())
        self.clear_gradients(self.fullyc8.params())

        out1 = self.conv1.forward(X)
        out2 = self.relu2.forward(out1)
        out3 = self.maxpool3.forward(out2)
        out4 = self.conv4.forward(out3)
        out5 = self.relu5.forward(out4)
        out6 = self.maxpool6.forward(out5)
        out7 = self.flatten7.forward(out6)
        out8 = self.fullyc8.forward(out7)

        loss_reg_FC, grad_reg_FC = l2_regularization(self.fullyc8.params()['W'].value, self.reg)

        loss, gradient = softmax_with_cross_entropy(out8, y)

        loss += loss_reg_FC

        gradient8 = self.fullyc8.backward(gradient)
        
        self.fullyc8.params()['W'].grad += grad_reg_FC
        
        gradient7 = self.flatten7.backward(gradient8)
        gradient6 = self.maxpool6.backward(gradient7)
        gradient5 = self.relu5.backward(gradient6)
        gradient4 = self.conv4.backward(gradient5)
        gradient3 = self.maxpool3.backward(gradient4)
        gradient2 = self.relu2.backward(gradient3)
        gradient1 = self.conv1.backward(gradient2)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        out1 = self.conv1.forward(X)
        out2 = self.relu2.forward(out1)
        out3 = self.maxpool3.forward(out2)
        out4 = self.conv4.forward(out3)
        out5 = self.relu5.forward(out4)
        out6 = self.maxpool6.forward(out5)
        out7 = self.flatten7.forward(out6)
        out8 = self.fullyc8.forward(out7)

        pred = np.argmax(out8, axis=1)

        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        conv1params = self.conv1.params()
        result['W1'] = conv1params['W']
        result['B1'] = conv1params['B']

        conv4params = self.conv4.params()
        result['W2'] = conv4params['W']
        result['B2'] = conv4params['B']

        FC8params = self.fullyc8.params()
        result['W3'] = FC8params['W']
        result['B3'] = FC8params['B']

        return result
          

