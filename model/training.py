from data import DataHandler

from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    def check_model(klass, name, regressor):
        (X, y), (X_test, y_test) = DataHandler.load_train_test()

        regressor.fit(X, y)
        return klass._check(regressor, name, X, y, X_test, y_test)

    @classmethod
    def _check(klass, estimator, name, X, y, X_test, y_test):
        predictions = estimator.predict(X_test)
        X = estimator.predict(X)
        print
        print '============== {0} ================='.format(name)
        print 'On train set MAE {0}'.format(mean_absolute_error(X, y))
        print 'On train set MSE {0}'.format(mean_squared_error(X, y))

        print 'On test set MAE {0}'.format(mean_absolute_error(predictions, y_test))
        print 'On test set MSE {0}'.format(mean_squared_error(predictions, y_test))
        print '======================================'

        plt.figure()
        # plt.subplot(1, 2, 1)
        plt.grid(True)
        plt.scatter(y, X, alpha=0.5, color='red', label='training data')
        plt.scatter(y_test, predictions, alpha=0.5,
                    color='blue', label='test data')
        plt.legend()
        plt.title('Input/Output correlation for {0}'.format(name))
        plt.axes().set_aspect('equal', 'datalim')
        plt.xlabel('true values')
        plt.ylabel('predicted values')
        plt.show()
        print 'On train set MSE {0}'.format(mean_squared_error(X, y))
        return mean_absolute_error(predictions, y_test)

    @classmethod
    def search(klass, name, regressor, parameters):
        print "Tuning the parameters.\nAll available:"
        for k in regressor.get_params().keys():
            print k
        (X, y), (X_test, y_test) = DataHandler.load_train_test()

        grid = GridSearchCV(regressor, parameters, cv=3, verbose=1, n_jobs=-1)
        grid.fit(X, y)
        print 'Best parameters', grid.best_params_

        # Now look at the best estimator
        klass._check(grid.best_estimator_, name, X, y, X_test, y_test)
