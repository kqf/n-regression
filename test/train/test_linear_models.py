import pytest

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from model.training import Trainer as ttr


@pytest.mark.onlylocal
def test_linear_model(data):
    ttr.check_model(data, 'SGD',
                    make_pipeline(
                        # StandardScaler(),
                        PolynomialFeatures(),
                        SGDRegressor()
                    ))
