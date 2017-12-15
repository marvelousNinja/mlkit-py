import numpy as np

def make_scaler(X):
  mean = np.mean(X, axis=0)
  std = np.std(X, axis=0)
  return lambda X: (X - mean) / std
