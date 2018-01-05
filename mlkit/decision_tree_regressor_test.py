import pytest
from mlkit.decision_tree_regressor import DecisionTreeRegressor
from mlkit.metrics import mae
from mlkit.datasets import load_boston

def test_on_boston():
    X, y = load_boston()
    model = DecisionTreeRegressor()
    model.fit(X[:400], y[:400])
    y_pred = model.predict(X[400:])
    assert mae(y[400:], y_pred) == pytest.approx(4.0, 0.1)
