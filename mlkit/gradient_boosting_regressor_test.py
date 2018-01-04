import os
import pytest
import pandas as pd

from mlkit.gradient_boosting_regressor import GradientBoostingRegressor
from mlkit.metrics import mae

def test_on_iris():
    full_path = os.path.realpath(__file__)
    dir_name = os.path.dirname(full_path)
    iris = pd.read_csv(dir_name + '/datasets/boston_house_prices.csv')

    X = iris.drop('MEDV', axis=1).values
    y = iris['MEDV'].values

    model = GradientBoostingRegressor()
    model.fit(X[:400], y[:400])
    y_pred = model.predict(X[400:])
    assert mae(y[400:], y_pred) == pytest.approx(4.3, 0.1)
