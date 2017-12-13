import numpy as np

class KnnRegressor():
  def fit(self, X, y):
    self.X = np.array(X)
    self.y = np.array(y)
    self.distance = lambda a, b: np.sqrt(np.sum((a - b) ** 2))

  def predict(self, X):
    predictions = []

    for x in X:
      distances = [self.distance(x, _x) for _x in self.X]
      order = np.argsort(distances)
      nearest = self.y[order][-5:]
      predictions.append(np.mean(nearest))
    return predictions


def test_init():
  KnnRegressor()

def test_fit():
  model = KnnRegressor()
  model.fit([[0, 0], [1, 1]], [1, 2])

def test_predict():
  model = KnnRegressor()
  model.fit([0], [1])
  assert model.predict([0]) == [1.0]
