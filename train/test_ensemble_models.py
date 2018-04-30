import unittest
from model.training import Trainer as ttr
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor


class TestEnsembleModels(unittest.TestCase):

    # TODO: Add separate tests for ensemble regressors
    #
    def test_random_forest(self):
        ttr.check_model('Random Forest', RandomForestRegressor())

    @unittest.skip('')
    def test_ada(self):
        ttr.check_model('Ada boost', AdaBoostRegressor())

    @unittest.skip('')
    def test_gradient(self):
        ttr.check_model('Gradient boost', GradientBoostingRegressor())

    # @unittest.skip('')
    def test_bagging(self):
        ttr.check_model('Bagging', BaggingRegressor())

    @unittest.skip('')
    def test_lgbm(self):
        from lightgbm import LGBMRegressor
        ttr.check_model('LGBM', LGBMRegressor())
