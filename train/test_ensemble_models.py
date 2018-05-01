import pytest
from model.training import Trainer as ttr
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor


# NB: These models are too heavy, but still it was interesting to check


@pytest.mark.skip('These models are too heavy')
def test_random_forest(self):
    ttr.check_model('Random Forest', RandomForestRegressor())


@pytest.mark.skip('These models are too heavy')
def test_ada(self):
    ttr.check_model('Ada boost', AdaBoostRegressor())


@pytest.mark.skip('These models are too heavy')
def test_gradient(self):
    ttr.check_model('Gradient boost', GradientBoostingRegressor())


@pytest.mark.skip('These models are too heavy')
def test_bagging(self):
    ttr.check_model('Bagging', BaggingRegressor())
