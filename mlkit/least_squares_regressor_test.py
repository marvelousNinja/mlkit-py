import pytest
from mlkit.least_squares_regressor import LeastSquaresRegressor
from mlkit.metrics import mae
from mlkit.util import make_scaler
from mlkit.datasets import load_boston

def test_on_iris():
    X, y = load_boston()
    scale = make_scaler(X[:400])
    model = LeastSquaresRegressor()
    model.fit(scale(X[:400]), y[:400])
    y_pred = model.predict(scale(X[400:]))
    assert mae(y[400:], y_pred) == pytest.approx(5, 0.1)
