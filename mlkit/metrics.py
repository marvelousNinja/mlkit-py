import numpy as np

def mse(y_true, y_pred):
  return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
  return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
  return np.mean(np.absolute(y_true - y_pred))

def rmsle(y_true, y_pred):
  return np.sqrt(np.mean((np.log(y_pred + 1) - np.log(y_true + 1)) ** 2))
