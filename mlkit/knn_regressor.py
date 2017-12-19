import numpy as np

class KnnRegressor():
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.distance = lambda a, b: np.sqrt(np.sum((a - b) ** 2))
        self.n = 5

    def predict(self, X):
        predictions = []

        for x in X:
            distances = [self.distance(x, _x) for _x in self.X]
            order = np.argsort(distances)
            nearest = self.y[order][:self.n]
            predictions.append(np.mean(nearest))
        return predictions
