import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(W ** 2)    
    grad = W * 2 * reg_strength

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (N, batch_size) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    predictions_copy = predictions.copy()

    if predictions_copy.ndim == 1 :
        predictions_copy -= np.max(predictions_copy)
        predictions_copy = np.exp(predictions_copy)
        
        divisor = np.sum(predictions_copy)

        probs = predictions_copy / divisor

        loss = -np.log(probs[target_index])

        dprediction = probs
        dprediction[target_index] -= 1
    else:
        predictions_copy -= np.max(predictions_copy, axis=1, keepdims=True)
        predictions_copy = np.exp(predictions_copy)

        divisor = np.sum(predictions_copy, axis=1, keepdims=True)

        probs = predictions_copy / divisor

        loss = np.mean(-np.log(probs[range(probs.shape[0]), target_index]))

        batch_size = target_index.shape[0]
        
        dprediction = probs
        dprediction[range(batch_size), target_index] -= 1
        dprediction /= batch_size

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X

        return np.maximum(X, 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops

        d_result = d_out
        d_result[self.X <= 0] = 0

        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.01 * np.random.randn(n_input, n_output))
        self.B = Param(0.01 * np.random.randn(1, n_output))

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        result = X.dot(self.W.value) + self.B.value

        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        dW = self.X.T.dot(d_out)
        self.W.grad = self.W.grad + dW
        
        dB = np.sum(d_out, axis=0, keepdims=True) 
        self.B.grad = self.B.grad + dB

        d_input = d_out.dot(self.W.value.T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}



    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.padding = padding
        
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )
        self.B = Param(np.zeros(out_channels))
        
        self.X = None


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # Padding 
        X = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')        
        self.X = X

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        out_width = int((width - self.filter_size + 2 * self.padding) + 1)
        out_height = int((height - self.filter_size + 2 * self.padding) + 1)

        result = np.zeros((batch_size, out_height, out_width, self.out_channels))

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops 
        stretched_W = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)

        for y in range(out_height):
            for x in range(out_width):
                stretched_X = np.reshape(X[:, y : y + self.filter_size, x : x + self.filter_size, :], (batch_size, -1))
                result[:, y, x, :] = np.dot(stretched_X, stretched_W) + self.B.value

        return result        		 
        

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, in_channels = self.X.shape
        #there is batch size 
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # MY_TODO : rename variables 


        d_output = np.zeros_like(self.X)
        stretched_W = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)
        local_d_out = np.zeros((batch_size, out_channels))
    
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                local_X = self.X[:, y : y + self.filter_size, x : x + self.filter_size, :].reshape(batch_size, -1)

                local_dO = d_out[ :, y, x, : ].reshape(batch_size, out_channels)

                local_dW = np.dot(local_X.T, local_dO).reshape(self.filter_size, 
                                                               self.filter_size, 
                                                               self.in_channels, 
                                                               self.out_channels)

                local_d_output = np.dot(local_dO, stretched_W.T).reshape(batch_size,
                                                                         self.filter_size, 
                                                                         self.filter_size, 
                                                                         self.in_channels) 
                
                d_output[:, y : y + self.filter_size, x : x + self.filter_size, :] += local_d_output

                self.W.grad += local_dW
                self.B.grad += np.sum(local_dO, axis=0)

        #print(self.W.grad)

        if self.padding != 0:
            return d_output[:, self.padding : -self.padding, self.padding : -self.padding, :]
        else:
            return d_output

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_width = int((width - self.pool_size) / self.stride + 1)
        out_height = int((height - self.pool_size) / self.stride + 1)

        result = np.zeros((batch_size, out_height, out_width, channels))

        for y in range(out_height):
            for x in range(out_width):
                stretched_X = np.reshape(X[:, y * self.stride : y * self.stride + self.pool_size, 
                                           x * self.stride : x * self.stride + self.pool_size, :], (batch_size, -1, channels))
                
                result[:, y, x, :] = np.max(stretched_X, axis=1)

        return result        

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape

        d_input = np.zeros_like(self.X)

        # MY_TODO : Refactor this shit, if u can 

        for y in range(out_height):
            for x in range(out_width):    
                stretched_X = np.reshape(self.X[:, y * self.stride : y * self.stride + self.pool_size, 
                                           x * self.stride : x * self.stride + self.pool_size, :], (batch_size, -1, channels))    

                stretched_d_input = np.zeros((batch_size, self.pool_size * self.pool_size, channels))    

                for bs in range(batch_size):
                    for channel in range(channels):
                        idx = np.argmax(stretched_X[bs, :, channel])
                        stretched_d_input[bs, idx, channel] += d_out[bs, y, x, channel]

                stretched_d_input = stretched_d_input.reshape(batch_size, self.pool_size, self.pool_size, channels)
                d_input[:, y * self.stride : y * self.stride + self.pool_size, 
                                           x * self.stride : x * self.stride + self.pool_size, :] += stretched_d_input

        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        result = X.reshape(self.X_shape[0], -1)

        return result

    def backward(self, d_out):
        # TODO: Implement backward pass
        d_out = d_out.reshape(self.X_shape[0], self.X_shape[1], self.X_shape[2], self.X_shape[3])

        return d_out

    def params(self):
        # No params!
        return {}
