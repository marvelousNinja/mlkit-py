import os
import pytest
import pandas as pd

from .normal_equation_regressor import NormalEquationRegressor
from .metrics import mae
from .util import make_scaler

def test_on_iris():
    full_path = os.path.realpath(__file__)
    dir_name = os.path.dirname(full_path)
    iris = pd.read_csv(dir_name + '/datasets/boston_house_prices.csv')

    X = iris.drop('MEDV', axis=1).values
    y = iris['MEDV'].values

    scale = make_scaler(X[:400])

    model = NormalEquationRegressor()

    model.fit(scale(X[:400]), y[:400])
    y_pred = model.predict(scale(X[400:]))
    assert mae(y[400:], y_pred) == pytest.approx(5, 0.1)
