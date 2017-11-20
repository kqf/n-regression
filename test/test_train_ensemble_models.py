import unittest
from model.training import Trainer as ttr 


class TestEnsembleModels(unittest.TestCase):

    # TODO: Add separate tests for ensemble regressors
    #
    @unittest.skip('')
    def test_random_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        ttr.check_model('Random Forest', RandomForestRegressor())

    def test_ada(self):
        from sklearn.ensemble import AdaBoostRegressor
        ttr.check_model('Ada boost', AdaBoostRegressor())

    def test_gradient(self):
        from sklearn.ensemble import GradientBoostingRegressor
        ttr.check_model('Gradient boost', GradientBoostingRegressor())

