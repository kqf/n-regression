import unittest
from model.training import Trainer as ttr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, Normalizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.neural_network import MLPRegressor



class TestCompositeModels(unittest.TestCase):

    # @unittest.skip('')
    def test_ridge(self):
        ttr.check_model(
            'MLPRegressor',
            MLPRegressor(hidden_layer_sizes=(50, 15,))
        )