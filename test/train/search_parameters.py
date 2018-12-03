import pytest

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from model.training import Trainer as ttr
from model.training import ColumnRemover


# This is a main working file it includes parameter tuning scenarios
#

@pytest.mark.onlylocal
def test_search_random_forest(data):

    grid_params = {
        "randomforestregressor__n_estimators": [20, 50, 100, 150, 200],
        "randomforestregressor__criterion": ["mse"],
        "randomforestregressor__min_samples_leaf": [2, 4, 8, 16, 32],
    }
    model = make_pipeline(
        ColumnRemover(("timeStamp",)),
        RandomForestRegressor(),
    )

    ttr.search(data, "Random Forest", model, grid_params)
