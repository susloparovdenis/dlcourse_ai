import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    predictions = np.array(predictions)
    predictions = predictions - np.max(predictions, axis=-1, keepdims=True)

    exp_preds = np.exp(predictions)
    denom = np.sum(exp_preds, axis=-1, keepdims=True)
    return exp_preds / denom


def get_one_hot(target_index, num_classes):
    target_index = np.array(target_index)
    one_hot = np.identity(num_classes)[target_index.flat]
    if one_hot.ndim > 2:
        return one_hot[0]
    return one_hot.squeeze()


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops

    probs = np.array(probs)
    num_classes = probs.shape[-1]
    one_hot = get_one_hot(target_index, num_classes)

    return np.mean(np.sum(-np.log(probs) * one_hot, axis=-1))


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # TODO implement softmax with cross-entropy
    pred_probs = softmax(predictions)
    batch_size = pred_probs.shape[0] if pred_probs.ndim > 1 else 1
    loss = cross_entropy_loss(pred_probs, target_index)

    num_classes = predictions.shape[-1]
    dprediction = pred_probs - get_one_hot(target_index, num_classes)
    dprediction = dprediction / batch_size
    # print(loss, dprediction)
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = np.sum(np.power(W, 2)) * reg_strength
    grad = 2 * W * reg_strength

    return loss, grad


def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    loss, grad = softmax_with_cross_entropy(predictions, target_index)

    dW = np.dot(X.T, grad)
    # Your final implementation shouldn't have any loops

    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            X_batches = X[batches_indices, :]
            y_batches = y.copy().reshape([-1, 1])[batches_indices, :]

            for i in range(len(X_batches)):
                bx, by = X_batches[i], y_batches[i]
                loss, dW = linear_softmax(bx, self.W, by)
                l2_loss, l2_dw = l2_regularization(self.W, reg)
                self.W = self.W - learning_rate * (dW+l2_dw)
                loss += l2_loss

            print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)
        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        pred = np.dot(X, self.W)
        y_pred = np.argmax(pred, axis=1)
        return y_pred
