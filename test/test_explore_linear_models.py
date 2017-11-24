import unittest
from model.training import Trainer as ttr 
from sklearn.svm import SVR 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge 
from sklearn.linear_model import LassoLars 
from sklearn.linear_model import ElasticNet 
from sklearn.linear_model import SGDRegressor


class TestExploreLinearModel(unittest.TestCase):

    def test_linear(self):
        models = {
            'SGD': SGDRegressor(),
            'Lasso': Lasso(),
            'Ridge': Ridge(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=1),
            'SVR': SVR(kernel='poly', degree=4),
            'LassoLars': LassoLars()
        }

        scores = [ttr.check_model(*kv) for kv in models.iteritems()]
