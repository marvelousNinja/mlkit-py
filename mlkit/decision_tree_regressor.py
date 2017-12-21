import numpy as np

# TODO AS: Yeah, sure
def get_mean_split(column, X, y):
    return [np.mean(X[:, column])]

def get_random_split(column, X, y):
    values = X[:, column]
    min, max = np.min(values), np.max(values)
    point = np.random.uniform(min, max)
    return [point]

def mse_impurity(y):
    mean = np.mean(y)
    return np.mean((y - mean) ** 2)

# TODO AS: Categorical split needs equality?
def split_on_value(column, value):
    return lambda X: X[:, column] >= value

class DecisionTreeRegressor():
    def fit(self, X, y):
        self.root = make_node(
            X, y,
            leaf_predictor=np.mean,
            # TODO AS: All these needed for split only
            get_split_candidates=get_mean_split,
            impurity=mse_impurity,
            make_split=split_on_value)

    def predict(self, X):
        return self.root(X)

def make_node(X, y, leaf_predictor, get_split_candidates, impurity, make_split):
    split = find_best_split(
        X, y,
        get_split_candidates=get_split_candidates,
        impurity=impurity,
        make_split=make_split)
    if not split:
        leaf_value = leaf_predictor(y)
        return lambda X: leaf_value

    mask = split(X)
    X_right, y_right = X[mask], y[mask]
    X_left, y_left = X[~mask], y[~mask]

    right = make_node(X_right, y_right, leaf_predictor, get_split_candidates, impurity, make_split)
    left = make_node(X_left, y_left, leaf_predictor, get_split_candidates, impurity, make_split)

    def decision(X):
        predictions = np.zeros(len(X))
        mask = split(X)
        predictions[mask] = right(X[mask])
        predictions[~mask] = left(X[~mask])
        return predictions

    return decision

def find_best_split(X, y, get_split_candidates, impurity, make_split):
    # TODO AS: Criterion needs to be parametrized
    # TODO AS: Criterion - the less it is => the better?
    min_leaf_samples = 5
    curr_value = impurity(y)
    curr_split = None

    for column in range(len(X[0])):
      # TODO AS: Split point generator needs to be parametrized
        for value in get_split_candidates(column, X, y):
            # TODO AS: Split condition generator needs to pe parametrized
            split = make_split(column, value)
            mask = split(X)
            y_right = y[mask]
            y_left = y[~mask]

            if len(y_right) < min_leaf_samples or len(y_left) < min_leaf_samples:
                continue

            # TODO AS: Min leaf values should be encorporated here
            split_value = (impurity(y_right) * len(y_right) + impurity(y_left) * len(y_left)) / len(y)
            # TODO AS: Min gain should be encorporated here
            if (split_value < curr_value):
                curr_split = split
                curr_value = split_value

    return curr_split
