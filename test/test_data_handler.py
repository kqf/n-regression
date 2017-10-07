import unittest


from model.data import DataHandler
from matplotlib import pyplot as plt
import seaborn



class TestDataHandler(unittest.TestCase):


    def test_loads_data(self):
        data = DataHandler.load()
        self.assertIsNotNone(data)

        # TODO: Add assert raises if empty file is passed
        print data.info()


    def test_loads_train_data(self):
        (X, y), (X_test, y_test) = DataHandler.load_train_test()

        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_test)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.hist(y)
        plt.title('train data distribution')

        plt.subplot(1, 2, 2)
        plt.hist(y_test)
        plt.title('test data distribution') 
        # plt.savefig('target_distribution_in_input_and_output.png')
        plt.show()
        print X.info()

