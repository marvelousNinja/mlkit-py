import numpy as np

def euclidean(A, B):
    return np.sqrt(-2 * (A @ B.T) + np.sum(A ** 2, axis=1).reshape(-1, 1) + np.sum(B ** 2, axis=1))

class KnnRegressor():
    def __init__(self, k_neighbours=5, distance=euclidean):
        self.X_train = None
        self.y_train = None
        self.k_neighbours = k_neighbours
        self.distance = distance

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        distances = self.distance(X, self.X_train)
        neighbour_indicies = np.argsort(distances, axis=1)[:, :self.k_neighbours]
        return [np.mean(self.y_train[neighbour_indicies[i]]) for i in range(len(X))]
