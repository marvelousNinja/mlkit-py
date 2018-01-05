import os
import pandas as pd

def _load_csv(filename):
    full_path = os.path.realpath(__file__)
    dir_name = os.path.dirname(full_path)
    return pd.read_csv(dir_name + '/' + filename)

def load_boston():
    dataframe = _load_csv('boston_house_prices.csv')
    X = dataframe.drop('MEDV', axis=1).values
    y = dataframe['MEDV'].values
    return X, y
