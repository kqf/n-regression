import unittest
from model.training import Trainer as ttr
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class TestCompositeModels(unittest.TestCase):

    # @unittest.skip('')
    def test_ridge(self):
        estimator = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(50, 15,))
        )

        ttr.check_model('MLPRegressor', estimator)

    def test_scan(self):
        nsize = "mlpregressor__hidden_layer_sizes"
        parameters = {
            nsize: [(16,), (16, 8), (32, 1)]
        }

        svr = make_pipeline(
            StandardScaler(),
            MLPRegressor(),
        )
        ttr.search('SVR ', svr, parameters)
