import pytest
from mlkit.linear_regressor import LinearRegressor
from mlkit.metrics import mae
from mlkit.util import make_scaler
from mlkit.datasets import load_boston

def test_on_boston():
    X, y = load_boston()
    scale = make_scaler(X[:400])
    model = LinearRegressor()
    model.fit(scale(X[:400]), y[:400])
    y_pred = model.predict(scale(X[400:]))
    assert mae(y[400:], y_pred) == pytest.approx(5, 0.1)
