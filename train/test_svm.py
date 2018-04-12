import unittest
from model.training import Trainer as ttr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class TestSVM(unittest.TestCase):

    def test_rbf(self):
        from sklearn.svm import SVR
        # TODO: Try grid search on different kernels
        #
        kernels = [
            "poly",
            "sigmoid",
            "rbf",
        ]
        for kernel in kernels:
            svr = make_pipeline(
                # PolynomialFeatures(),
                StandardScaler(),
                SVR(kernel=kernel, C=25),
            )
            ttr.check_model('SVR ' + kernel, svr)
