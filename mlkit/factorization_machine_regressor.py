import numpy as np

class FactorizationMachineRegressor():
    def __init__(self, k):
        # TODO AS: Assume that degree is 2
        self.d = 2
        self.k = k
        self.W = None
        self.V = None
        self.bias = 0

    def fit(self, X, y):
        n_features = len(X[0])
        self.W = np.random.uniform(-1, 1, n_features)
        self.V = np.random.uniform(-1, 1, (n_features, self.k))

        # TODO AS: And we somehow fit it...
        pass

    def predict(self, X):
        # TODO AS: Generalize for k and d
        predictions = self.bias + X @ self.W
        for i in range(self.k):
            V = self.V[:, i]
            predictions += ((X @ V) ** 2 - ((X ** 2) @ (V ** 2))) / 2
        return predictions
