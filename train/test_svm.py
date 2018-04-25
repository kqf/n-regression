import unittest
from model.training import Trainer as ttr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class TestSVM(unittest.TestCase):

    @unittest.skip('')
    def test(self):
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

    def test_scan(self):
        parameters = {
            "svr__kernel": ["rbf"],
            # "kernel": ["rbf", "poly", "sigmoid"],
            "svr__C": [1, 10, 25, 30],
            "svr__gamma": ["auto", 2. / 16, 4. / 16, 1.]
        }

        svr = make_pipeline(
            StandardScaler(),
            SVR(),
        )
        ttr.search('SVR ', svr, parameters)
