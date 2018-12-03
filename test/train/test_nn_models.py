import pytest
from model.training import Trainer as ttr
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@pytest.mark.onlylocal
def test_ridge(data):
    estimator = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(50, 15,))
    )

    ttr.check_model(data, "MLPRegressor", estimator)
