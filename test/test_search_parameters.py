import unittest

from model.data import DataHandler
from model.training import Trainer as ttr 

class TestSearchParameters(unittest.TestCase):


    # This is a main working file it includes parameter tuning scenarios
    #

    def test_search_random_forest(self):
        from sklearn.ensemble import RandomForestRegressor

        grid_params = {
            'randomforestregressor__n_estimators': [20, 50, 100, 150, 200],
            'randomforestregressor__criterion': ['mse'],
            'randomforestregressor__min_samples_leaf': [2, 4, 8, 16, 32],
        }

        ttr.search('Random Forest', RandomForestRegressor(n_jobs = -1), grid_params)


