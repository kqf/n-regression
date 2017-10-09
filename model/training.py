from data import DataHandler

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from matplotlib import pyplot as plt
import seaborn


# Avoid sklearn version mismatch
try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    from sklearn.grid_search import GridSearchCV


class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()

    # NB: Now it's easy to extend the training proces 
    #     by taking adwantage of iheritance. It can be useful in future
    #

    @classmethod
    def pipeline(klass, regressor):
        return make_pipeline(StandardScaler(), regressor)


    @classmethod
    def check_model(klass, name, regressor):
        (X, y), (X_test, y_test) = DataHandler.load_train_test()

        estimator = klass.pipeline(regressor)
        estimator.fit(X, y)
        klass._check(estimator, name, X, y, X_test, y_test)


    @classmethod
    def _check(klass, estimator, name, X, y, X_test, y_test):
        predictions = estimator.predict(X_test)
        print '{0} MAE {1}'.format(name, mean_absolute_error(predictions, y_test))
        print '{0} MSE {1}'.format(name, mean_squared_error(predictions, y_test))

        plt.figure()
        # plt.subplot(1, 2, 1)
        plt.grid(True)
        plt.scatter(y, estimator.predict(X), alpha = 0.5, color = 'red', label = 'training data')
        plt.scatter(y_test, predictions, alpha = 0.5, color = 'blue', label = 'test data')
        plt.legend()
        plt.title('Input/Output correlation for {0}'.format(name))
        plt.axes().set_aspect('equal')
        plt.xlabel('true values')
        plt.ylabel('predicted values')
        plt.show()


    @classmethod   
    def search(klass, name, regressor, parameters):
        (X, y), (X_test, y_test) = DataHandler.load_train_test()

        estimator = klass.pipeline(regressor)
        grid = GridSearchCV(estimator, parameters, cv = 3)
        grid.fit(X, y)


        print 'Best parameters', grid.best_params_


        # Now look at the best estimator
        klass._check(grid.best_estimator_, name, X, y, X_test, y_test)
