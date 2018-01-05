import pytest
from mlkit.gradient_boosting_regressor import GradientBoostingRegressor
from mlkit.metrics import mae
from mlkit.datasets import load_boston

def test_on_boston():
    X, y = load_boston()
    model = GradientBoostingRegressor()
    model.fit(X[:400], y[:400])
    y_pred = model.predict(X[400:])
    assert mae(y[400:], y_pred) == pytest.approx(4.3, 0.1)
