import unittest
from model.training import Trainer as ttr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, Normalizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import SGDRegressor, Ridge, Lasso, LinearRegression



def transformer(klass):
    class ClassTransformer(klass, TransformerMixin):

        def transform(self, X, *_):
            result = self.predict(X)
            return result.reshape(-1, 1)

    return ClassTransformer


class TestCompositeModels(unittest.TestCase):

    def test_composite(self):
        estimator = make_union(
            transformer(Ridge)(alpha=0.01),
            transformer(Lasso)(),
            transformer(SGDRegressor)()
        )

        pipeline = make_pipeline(
            PolynomialFeatures(degree=5),
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