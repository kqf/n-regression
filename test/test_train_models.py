import unittest
from model.training import Trainer as ttr 

class TestDifferentModles(unittest.TestCase):

    # It's clear that the data is not linear
    # but let's try the simplest approach
    #

    # @unittest.skip('')
    def test_linear(self):
        from sklearn.linear_model import SGDRegressor
        ttr.check_model('SGD', SGDRegressor(max_iter=5 ))


    def test_random_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        ttr.check_model('Random Forest', RandomForestRegressor())


