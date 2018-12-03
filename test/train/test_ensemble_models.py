import pytest
from model.training import Trainer as ttr
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor


# NB: These models are too heavy, but still it was interesting to check


@pytest.mark.skip("These models are too heavy")
def test_random_forest(data):
    ttr.check_model(data, "Random Forest", RandomForestRegressor())


@pytest.mark.skip("These models are too heavy")
def test_ada(data):
    ttr.check_model(data, "Ada boost", AdaBoostRegressor())


@pytest.mark.skip("These models are too heavy")
def test_gradient(data):
    ttr.check_model(data, "Gradient boost", GradientBoostingRegressor())


@pytest.mark.skip("These models are too heavy")
def test_bagging(data):
    ttr.check_model(data, "Bagging", BaggingRegressor())
