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
