import pytest

from model.training import Trainer as ttr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@pytest.mark.skip('')
@pytest.mark.onlylocal
def test():
    kernels = [
        'poly',
        'sigmoid',
        'rbf',
    ]
    for kernel in kernels:
        svr = make_pipeline(
            # PolynomialFeatures(),
            StandardScaler(),
            SVR(kernel=kernel, C=25),
        )
        ttr.check_model('SVR ' + kernel, svr)


@pytest.mark.onlylocal
def test_scan():
    parameters = {
        'svr__kernel': ['rbf'],
        # 'kernel': ['rbf', 'poly', 'sigmoid'],
        'svr__C': [1, 10, 25, 30],
        'svr__gamma': ['auto', 2. / 16, 4. / 16, 1.]
    }

    svr = make_pipeline(
        StandardScaler(),
        SVR(),
    )
    ttr.search('SVR ', svr, parameters)
