import numpy as np
from mlkit.decision_tree_regressor import (
    make_node,
    get_all_columns,
    get_mean_split,
    mse_impurity,
    split_on_value)

class GradientBoostingRegressor():
    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        self.trees = []
        self.learning_rate = 0.1

        pred = np.full(len(y), self.initial_prediction)
        for i in range(10):
            negative_gradient = y - pred

            tree = make_node(
                X, negative_gradient,
                sample_columns=get_all_columns,
                leaf_predictor=np.mean,
                get_split_candidates=get_mean_split,
                impurity=mse_impurity,
                make_split=split_on_value)

            pred += self.learning_rate * tree(X)
            self.trees.append(tree)

    def predict(self, X):
        pred = np.full(len(X), self.initial_prediction)
        for tree in self.trees:
            pred += self.learning_rate * tree(X)
        return pred
