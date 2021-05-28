import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


class RegressionPrediction:
    def __init__(self, _x_src, _y_src, degree=2):
        self.x_src = _x_src
        self.y_src = _y_src
        lr = LinearRegression()
        pf = PolynomialFeatures(degree=degree)
        self.x_fit = pf.fit_transform(self.x_src)
        self.x_fit_tmp = np.arange(self.x_src.min(), self.x_src.max(), 1)[:, np.newaxis]
        lr.fit(self.x_fit, self.y_src)
        self.y_fit = lr.predict(pf.fit_transform(self.x_fit_tmp))

    @property
    def square_error(self):
        return mean_squared_error(self.y_src, self.y_fit)

    def plot(self):
        plt.scatter(self.x_src, self.y_src, label='Source points')
        plt.plot(self.x_fit, self.y_fit, label='Predict function', linestyle='--', color='orange')
        plt.legend(loc='upper left')
        plt.show()


def plot_prediction(x_src, y_src, x_fit, y_fit):
    plt.scatter(x_src, y_src, label='Source points')
    plt.plot(x_fit, y_fit, label='Prediction', linestyle='--', color='orange')
    plt.legend(loc='upper left')
    plt.show()


def predict_with_regression(x_src, y_src, degree=2):
    lr = LinearRegression()
    pf = PolynomialFeatures(degree=degree)
    x_quad = pf.fit_transform(x_src)
    x_fit = np.arange(x_src.min(), x_src.max(), 1)[:, np.newaxis]
    lr.fit(x_quad, y_src)
    y_quad_fit = lr.predict(pf.fit_transform(x_fit))
    return x_fit, y_quad_fit
