import numpy as np
from mlkit.decision_tree_regressor import (
    make_node,
    get_sqrt_random_columns,
    get_random_split,
    mse_impurity,
    split_on_value)

class ExtraTreesRegressor():
    def __init__(self, n_estimators=10, seed=0):
        self.n_estimators = n_estimators
        self.seed = seed
        self.trees = None

    def fit(self, X, y):
        np.random.seed(self.seed)
        self.trees = []
        for i in range(self.n_estimators):
            root = make_node(
                X, y,
                sample_columns=get_sqrt_random_columns,
                leaf_predictor=np.mean,
                get_split_candidates=get_random_split,
                impurity=mse_impurity,
                make_split=split_on_value)
            self.trees.append(root)

    def predict(self, X):
        acc = np.zeros(len(X))
        for tree in self.trees:
            acc = acc + tree(X)
        return acc / len(self.trees)
