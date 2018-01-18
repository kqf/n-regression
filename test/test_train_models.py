import unittest
from model.training import Trainer as ttr 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, Normalizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import SGDRegressor, Ridge, Lasso, LinearRegression


class RidgeTransformer(Ridge, TransformerMixin):

    def transform(self, X, *_):
        result = self.predict(X)
        return result.reshape(-1, 1)

class LassoTransformer(Lasso, TransformerMixin):

    def transform(self, X, *_):
        result = self.predict(X)
        return result.reshape(-1, 1)

class SGDRegressorTransformer(SGDRegressor, TransformerMixin):

    def transform(self, X, *_):
        result = self.predict(X)
        return result.reshape(-1, 1)

class TestCompositeModels(unittest.TestCase):

    def test_composite(self):
        estimator = make_union(
            RidgeTransformer(alpha=0.01),
            LassoTransformer(),
            SGDRegressorTransformer()
        )

        pipeline = make_pipeline(
            PolynomialFeatures(degree=4),
            Normalizer(),
            estimator,
            LinearRegression()
        )
        ttr.check_model('linear_composition', pipeline)


    @unittest.skip('')
    def test_lasso(self):
        ttr.check_model('Lasso', Lasso())


    @unittest.skip('')
    def test_ridge(self):
        ttr.check_model('Ridge', Ridge(alpha=0.1))


    @unittest.skip('')
    def test_elastic(self):
        from sklearn.linear_model import ElasticNet 
        ttr.check_model('ElasticNet', ElasticNet(alpha=1))


    @unittest.skip('')
    def test_svr(self):
        from sklearn.svm import SVR 
        # TODO: Try to search in differnt kernel functions
        #       rbf, polynomial
        ttr.check_model('SVR', SVR(kernel='poly', degree=4))

