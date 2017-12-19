import numpy as np

class NormalEquationRegressor():
    def fit(self, X, y):
      X = np.c_[X, np.ones(len(X))]
      self.w = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
      X = np.c_[X, np.ones(len(X))]
      return X @ self.w
