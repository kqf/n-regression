from model.data import DataHandler
from sklearn.datasets import make_regression
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pytest
import os


def make_data():
    # Generate slightly simpler linear model
    #

    X, y = make_regression(n_features=13)
    inputs, targets = pd.DataFrame(X), pd.DataFrame(y)

    # Generate fake time steps
    #
    inputs["timeStamp"] = [datetime(2018, 1, 1) + timedelta(hours=i)
                           for i in inputs.index]

    # Convert to the proper format
    inputs["timeStamp"] = inputs["timeStamp"].apply(
        lambda x: x.strftime("%Y/%m/%d-%H:%M:%S")
    )
    return inputs, targets


@pytest.fixture()
def datafiles():
    inputsf, targetsf = "data/inputs.csv", "data/targets.csv"
    if os.path.isfile(inputsf) and os.path.isfile(targetsf):
        return inputsf, targetsf

    # Generate and save the dataset
    inputs, targets = make_data()
    inputs.to_csv(inputsf, index=False)
    targets.to_csv(targetsf, index=False)
    return inputsf, targetsf


@pytest.fixture()
def data(datafiles):
    return DataHandler.load_train_test(*datafiles)
