import numpy as np

class LinearRegressor():
  def fit(self, X, y):
    X = np.c_[X, np.ones(len(X))]
    self.w = [0] * len(X[0])

    n_iter = 10000
    lr = 0.1

    for i in range(n_iter):
      grad = np.clip((2 * (X @ self.w - y) @ X), -0.1, 0.1)
      self.w = self.w - lr * grad

  def predict(self, X):
    X = np.c_[X, np.ones(len(X))]
    return X @ self.w
