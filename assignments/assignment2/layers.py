import numpy as np

from assignments.assignment1.linear_classifer import cross_entropy_loss, get_one_hot, softmax


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
    loss = np.sum(np.power(W, 2)) * reg_strength
    grad = 2 * W * reg_strength

    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    pred_probs = softmax(preds)
    batch_size = pred_probs.shape[0] if pred_probs.ndim > 1 else 1
    loss = cross_entropy_loss(pred_probs, target_index)

    num_classes = preds.shape[-1]
    dprediction = pred_probs - get_one_hot(target_index, num_classes)
    dprediction = dprediction / batch_size
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
        # Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        res = np.maximum(X,0)
        self.grad = X > 0
        
        return res

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
        # Implement backward pass
        # Your final implementation shouldn't have any loops
        
        d_result = self.grad *d_out
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        return np.dot(X, self.W.value) + self.B.value
      
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

        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)[None, :]
        return np.dot(d_out, self.W.value.T)
      

    def params(self):
        return {'W': self.W, 'B': self.B}
 