import numpy as np


def init_params(n_x, n_h, n_y):
    """
    Args:
        n_x: features
        n_h: hidden units
        n_y: classes

    Returns:
        params: initial parameters
    """
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros([n_h, 1])
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros([n_y, 1])

    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return params


def sigmoid(z):
    return 1/(1+np.exp(-z))


def forward_prop(X, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    Z1 = W1 @ X + b1
    A1 = np.tanh(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, cache


def avg_log_loss(A_last, Y):
    """
    Avg log-loss function
    reference: http://wiki.fast.ai/index.php/Log_Loss

    Args:
        A_last: last layer activations
        Y: true labels

    Returns:
        logloss
    """
    logprobs = Y * np.log(A_last) + (1-Y) * np.log(1-A_last)
    return np.mean(-logprobs)


def backward_prop(params, cache, X, Y):
    _, m = X.shape
    W1 = params['W1']
    W2 = params['W2']
    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = 1/m * dZ2 @ A1.T
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T @ dZ2 * (1-A1**2)
    dW1 = 1/m * dZ1 @ X.T
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return grads


def update_params(params, grads, learning_rate):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return params


def fit_params(X, Y, n_h, num_iterations=10000, learning_rate=1.2, debug=False):
    n_x, _ = X.shape
    n_y, _ = Y.shape
    params = init_params(n_x, n_h, n_y)

    for idx in range(num_iterations):

        A2, cache = forward_prop(X, params)

        cost = avg_log_loss(A2, Y)

        grads = backward_prop(params, cache, X, Y)

        params = update_params(params, grads, learning_rate)

        if debug and idx % 1000 == 0:
            print(f'Cost after iteration {idx}: {cost}')

    return params


def predict(params, X):
    A2, cache = forward_prop(X, params)
    return A2 > .5


class NeuralNet:
    def __init__(self, n_h=4, num_iterations=10000, learning_rate=1.2, debug=False):
        self.params = None
        self.kwargs = {
            'n_h': n_h,
            'num_iterations': num_iterations,
            'learning_rate': learning_rate,
            'debug': debug,
        }

    def fit(self, X, Y):
        self.params = fit_params(X, Y, **self.kwargs)

    def predict(self, X):
        return predict(self.params, X)
