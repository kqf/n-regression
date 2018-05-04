from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Avoid sklearn version mismatch
try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    from sklearn.grid_search import GridSearchCV


class Trainer():
    # NB: Now it's easy to extend the training proces
    #     by taking adwantage of iheritance. It can be useful in future
    #

    @classmethod
    def check_model(klass, data, name, regressor):
        X_tr, X_te, y_tr, y_te = data
        regressor.fit(X_tr, y_tr)

        y_te_pred = regressor.predict(X_te)
        y_tr_pred = regressor.predict(X_tr)
        print()
        print('============== {0} ================='.format(name))
        print('On train set MAE {0}'.format(
            mean_absolute_error(y_tr_pred, y_tr)))
        print('On train set MSE {0}'.format(
            mean_squared_error(y_tr_pred, y_tr)))
        print('On test set MAE {0}'.format(
            mean_absolute_error(y_te_pred, y_te)))
        print('On test set MSE {0}'.format(
            mean_squared_error(y_te_pred, y_te)))
        print('======================================')

        plt.figure()
        plt.grid(True)
        plt.scatter(y_tr, y_tr_pred, alpha=0.5,
                    color='red', label='training data')
        plt.scatter(y_te, y_te_pred, alpha=0.5,
                    color='blue', label='test data')
        plt.legend()
        plt.title('Input/Output correlation for {0}'.format(name))
        plt.axes().set_aspect('equal', 'datalim')
        plt.xlabel('true values')
        plt.ylabel('predicted values')
        plt.show()
        return mean_absolute_error(y_te_pred, y_te)

    @classmethod
    def search(klass, data, name, regressor, parameters):
        print('Tuning the parameters.\nAll available:')
        for k in regressor.get_params().keys():
            print(k)
        X_tr, X_te, y_tr, y_te = data
        # X_tr, X_te, y_tr, y_te =
        grid = GridSearchCV(regressor, parameters, cv=3,
                            verbose=1, n_jobs=-1)
        grid.fit(X_tr, y_tr)
        print('Best parameters', grid.best_params_)
        print('Best estimator Train RMS',
              mean_absolute_error(grid.best_estimator_.predict(X_tr), y_tr))
        print('Best estimator Test  RMS',
              mean_absolute_error(grid.best_estimator_.predict(X_te), y_te))
