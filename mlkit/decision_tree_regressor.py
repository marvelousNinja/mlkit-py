import numpy as np

class DecisionTreeRegressor():
  def fit(self, X, y):
    self.root = make_node(X, y)

  def predict(self, X):
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
    predictions[mask] = right(X[mask])
    predictions[~mask] = left(X[~mask])
    return predictions

  return decision

def find_best_split(X, y):
  # TODO AS: Criterion needs to be parametrized
  # TODO AS: Criterion - the less it is => the better?
  min_leaf_samples = 5
  curr_value = criterion(y)
  curr_split = None

  for column in range(len(X[0])):
    # TODO AS: Split point generator needs to be parametrized
    for value in get_split_candidates(column, X, y):
      # TODO AS: Split condition generator needs to pe parametrized
      split = make_split(column ,value)
      mask = split(X)
      y_right = y[mask]
      y_left = y[~mask]

      if len(y_right) < min_leaf_samples or len(y_left) < min_leaf_samples:
        continue

      # TODO AS: Min leaf values should be encorporated here
      split_value = (criterion(y_right) * len(y_right) + criterion(y_left) * len(y_left)) / len(y)
      # TODO AS: Min gain should be encorporated here
      if (split_value < curr_value):
        curr_split = split
        curr_value = split_value

  return curr_split

# TODO AS: Yeah, sure
def get_split_candidates(column, X, y):
  return [np.mean(X[:, column])]

def make_split(column, value):
  return lambda X: X[:, column] >= value

def criterion(y):
  mean = np.mean(y)
  return np.mean((y - mean) ** 2)
