import unittest
from model.training import Trainer as ttr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


class TestDifferentModles(unittest.TestCase):

    # It's clear that the data is not linear
    # but let's try the simplest approach
    #

    @unittest.skip('')
    def test_linear(self):
        from sklearn.linear_model import SGDRegressor
        ttr.check_model('SGD', SGDRegressor())

    @unittest.skip('')
    def test_lasso(self):
        from sklearn.linear_model import Lasso
        ttr.check_model('Lasso', Lasso())

    @unittest.skip('')
    def test_ridge(self):
        from sklearn.linear_model import Ridge
        ridge = make_pipeline([
            PolynomialFeatures(3),
            Ridge(alpha=0.1),
        ])
        ttr.check_model('Ridge', ridge)

    @unittest.skip('')
    def test_elastic(self):
        from sklearn.linear_model import ElasticNet
        ttr.check_model('ElasticNet', ElasticNet(alpha=1))

    def test_svr(self):
        from sklearn.svm import SVR
        # TODO: Try to search in differnt kernel functions
        #       rbf, polynomial
        ttr.check_model('SVR', SVR(gamma=0.1))

    @unittest.skip('')
    def test_lars(self):
        from sklearn.linear_model import LassoLars
        ttr.check_model('LassoLars', LassoLars())
