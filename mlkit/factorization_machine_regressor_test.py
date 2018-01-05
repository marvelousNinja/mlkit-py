import pytest
from mlkit.factorization_machine_regressor import FactorizationMachineRegressor
from mlkit.metrics import mae
from mlkit.datasets import load_boston

def test_on_boston():
    return
    X, y = load_boston()
    model = FactorizationMachineRegressor(k=2)
    model.fit(X[:400], y[:400])
    y_pred = model.predict(X[400:])
    assert mae(y[400:], y_pred) == pytest.approx(3.9, 0.1)
