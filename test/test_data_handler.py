from model.data import DataHandler
from matplotlib import pyplot as plt


def test_loads_data():
    data = DataHandler.load()
    assert data is not None

    # TODO: Add assert raises if empty file is passed
    print(data.info())


def test_loads_train_data():
    (X, y), (X_test, y_test) = DataHandler.load_train_test()

    assert X is not None
    assert y is not None
    assert X_test is not None
    assert y_test is not None

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(y)
    plt.title('train data distribution')
    plt.xlabel('target variable')
    plt.ylabel('counts')

    plt.subplot(1, 2, 2)
    plt.hist(y_test)
    plt.title('test data distribution')
    plt.xlabel('target variable')
    # plt.savefig('target_distribution_in_input_and_output.png')
    plt.show()
    print(X.info())
