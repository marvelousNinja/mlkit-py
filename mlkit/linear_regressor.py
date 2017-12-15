import numpy as np

class LinearRegressor():
  def fit(self, X, y):
    X = np.c_[X, np.ones(len(X))]

    self.w = np.random.normal(size=(len(X[0])), scale=0.5)
    cost = lambda X, y, w: np.mean(((X @ self.w - y) ** 2))
    grad = lambda X, y, w: 2 / len(X) * (X @ self.w - y) @ X

    n_iter = 1000
    lr = 0.1
    tol = 1e-8
    prev_cost = np.Inf

    for i in range(n_iter):
      self.w = self.w - lr * grad(X, y, self.w)
      curr_cost = cost(X, y, self.w)
      if (np.abs(prev_cost - curr_cost) < tol):
        break
      else:
        prev_cost = curr_cost

  def predict(self, X):
    X = np.c_[X, np.ones(len(X))]
    return X @ self.w
