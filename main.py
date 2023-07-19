import numpy as np


class LinearRegression:

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # function used for training
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # zero array with help of numpy
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T , (y_pred - y))  # deriv of J wrt w
            db = (1 / n_samples) * np.sum(y_pred - y)  # deriv of J wrt b

            # updating w and b simultaneously
            self.weights -= (self.lr * dw)
            self.bias -= (self.lr * db)

    # function used for inference
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred



