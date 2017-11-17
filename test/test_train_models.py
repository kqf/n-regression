import unittest
from model.training import Trainer as ttr 

class TestDifferentModles(unittest.TestCase):

    # It's clear that the data is not linear
    # but let's try the simplest approach
    #

    # @unittest.skip('')
    def test_linear(self):
        from sklearn.linear_model import SGDRegressor
        ttr.check_model('SGD', SGDRegressor())


    def test_lasso(self):
        from sklearn.linear_model import Lasso
        ttr.check_model('Lasso', Lasso())


    def test_ridge(self):
        from sklearn.linear_model import Ridge 
        ttr.check_model('Ridge', Ridge(alpha=0.1))


    def test_elastic(self):
        from sklearn.linear_model import ElasticNet 
        ttr.check_model('ElasticNet', ElasticNet(alpha=1))


    def test_svr(self):
        from sklearn.svm import SVR 
        # TODO: Try to search in differnt kernel functions
        #       rbf, polynomial
        ttr.check_model('SVR', SVR(kernel='poly', degree=4))



    # TODO: Add separate tests for ensemble regressors
    #
    
    @unittest.skip('')
    def test_random_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        ttr.check_model('Random Forest', RandomForestRegressor())



