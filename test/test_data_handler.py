from model.data import DataHandler
from matplotlib import pyplot as plt


def test_loads_data(datafiles):
    data = DataHandler.load(*datafiles)
    assert data is not None


def test_loads_train_data(datafiles):
    X_tr, X_te, y_tr, y_te = DataHandler.load_train_test(*datafiles)

    assert X_tr is not None
    assert X_te is not None
    assert y_tr is not None
    assert y_te is not None

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(y_te)
    plt.title('train data distribution')
    plt.xlabel('target variable')
    plt.ylabel('counts')

    plt.subplot(1, 2, 2)
    plt.hist(y_te)
    plt.title('test data distribution')
    plt.xlabel('target variable')
    # plt.savefig('target_distribution_in_input_and_output.png')
    print(X_tr.info())
