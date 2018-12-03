import pytest

from model.training import Trainer as ttr
from model.training import ColumnRemover
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@pytest.mark.onlylocal
def test(data):
    svr = make_pipeline(
        # PolynomialFeatures(),
        StandardScaler(),
        SVR(kernel="rbf", C=25, gamma="scale"),
    )
    ttr.check_model(data, "SVR rbf", svr)


@pytest.mark.skip("")
@pytest.mark.onlylocal
def test_scan(data):
    parameters = {
        "svr__kernel": ["rbf"],
        # "kernel": ["rbf", "poly", "sigmoid"],
        "svr__C": [1, 10, 25, 30],
        "svr__gamma": ["auto", 2. / 16, 4. / 16, 1.]
    }

    svr = make_pipeline(
        ColumnRemover(("timeStamp",)),
        StandardScaler(),
        SVR(),
    )
    ttr.search(data, "SVR ", svr, parameters)
