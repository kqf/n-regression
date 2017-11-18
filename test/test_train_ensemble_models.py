import unittest
from model.training import Trainer as ttr 


class TestDifferentModles(unittest.TestCase):

    # TODO: Add separate tests for ensemble regressors
    #
    @unittest.skip('')
    def test_random_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        ttr.check_model('Random Forest', RandomForestRegressor())

