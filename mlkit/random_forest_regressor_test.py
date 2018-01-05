import pytest
from mlkit.random_forest_regressor import RandomForestRegressor
from mlkit.metrics import mae
from mlkit.datasets import load_boston

def test_on_boston():
    X, y = load_boston()
    model = RandomForestRegressor()
    model.fit(X[:400], y[:400])
    y_pred = model.predict(X[400:])
    assert mae(y[400:], y_pred) == pytest.approx(3.2, 0.1)
