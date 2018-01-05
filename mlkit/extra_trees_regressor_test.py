import pytest
from mlkit.extra_trees_regressor import ExtraTreesRegressor
from mlkit.metrics import mae
from mlkit.datasets import load_boston

def test_on_boston():
    X, y = load_boston()
    model = ExtraTreesRegressor()
    model.fit(X[:400], y[:400])
    y_pred = model.predict(X[400:])
    assert mae(y[400:], y_pred) == pytest.approx(3.9, 0.1)
