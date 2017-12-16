import numpy as np

class DecisionTreeRegressor():
  def fit(X, y):
    self.root = make_node(X, y)

  def predict(X):
    return self.root(X)

def make_node(X, y):
  split = find_best_split(X, y)
  if not split:
    # TODO AS: Leaf predictor needs to be parametrized
    leaf_value = np.mean(y)
    return lambda X: leaf_value

  mask = split(X)
  X_right, y_right = X[mask], y[mask]
  X_left, y_left = X[~mask], y[~mask]

  right = make_node(X_right, y_right)
  left = make_node(X_left, y_left)

  def decision(X):
    predictions = np.zeros(len(X))
    mask = split(X)
    predictions[mask] = right(X)
    predictions[~mask] = left(X)
    return predictions

  return decision

def find_best_split(X, y):
  # TODO AS: Criterion needs to be parametrized
  # TODO AS: Criterion - the less it is => the better?
  curr_value = criterion(y)
  curr_split = None

  for column in len(X[0]):
    # TODO AS: Split point generator needs to be parametrized
    for value in np.unique(X[, column]):
      # TODO AS: Split condition generator needs to pe parametrized
      split = lambda X: X[, column] >= value
      mask = split(X)
      y_right = y[mask]
      y_left = y[~mask]

      split_value = (criterion(y_right) * len(y_right) + criterion(y_left) * len(y_left)) / len(y)
      if (split_value < curr_split):
        curr_split = split
        curr_value = split_value

  return curr_split
