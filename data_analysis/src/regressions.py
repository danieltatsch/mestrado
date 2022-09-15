from sklearn.linear_model import LinearRegression
from sklearn.metrics      import mean_squared_error, mean_absolute_error, r2_score

def get_MSE(y_test, y_pred, squared=True):
    return mean_squared_error(y_test, y_pred, squared=squared)

def get_RMSE(y_test, y_pred):
    return get_MSE(y_test, y_pred, False)

def get_r2_score(y_test, y_pred):
    return r2_score(y_test, y_pred)

def get_MAE(y_test, y_pred):
    return mean_absolute_error(y_test, y_pred)

class simple_linear_regression:
    def __init__(self, x, y):
        self.model = LinearRegression().fit(x, y)
        self.x = x
        self.y = y
        
    def get_coefficients(self):
        b0 = self.model.intercept_
        b1 = self.model.coef_

        return b0,b1

    def get_r2(self):
        return self.model.score(self.x, self.y)

    def get_predict_data(self, x_test):
        y_pred = self.model.predict(x_test)

        return y_pred