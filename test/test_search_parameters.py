import unittest

from model.data import DataHandler
from model.training import Trainer as ttr 

class TestSearchParameters(unittest.TestCase):


    # This is a main working file it includes parameter tuning scenarios
    #

    def test_search_random_forest(self):
        from sklearn.ensemble import RandomForestRegressor

        grid_params = {
            'randomforestregressor__n_estimators': [5, 10, 20, 50, 100],
            'randomforestregressor__criterion': ['mse', 'mae'],
            'randomforestregressor__min_samples_leaf': [0.001, 0.1, 0.2, 0.4],
        }

        ttr.search('Random forest', RandomForestRegressor(n_jobs = -1), grid_params)


