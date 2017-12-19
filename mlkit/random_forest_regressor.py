import numpy as np
from .decision_tree_regressor import make_node

class RandomForestRegressor():
    def fit(self, X, y):
        self.trees = []
        ids = range(len(X))
        for i in range(10):
            sample_ids = np.random.choice(ids, len(ids))
            X_sample, y_sample = X[sample_ids], y[sample_ids]
            root = make_node(X_sample, y_sample)
            self.trees.append(root)

    def predict(self, X):
        acc = np.zeros(len(X))
        for tree in self.trees:
            acc = acc + tree(X)
        return acc / len(self.trees)
