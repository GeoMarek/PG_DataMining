import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


class RegressionPrediction:
    def __init__(self, x_src, y_src, degree=2):
        self.x_src = x_src
        self.y_src = y_src
        self.degree = degree
        self.x_fit = np.arange(self.x_src.min(), self.x_src.max()+1, 1)[:, np.newaxis]
        poly_regr = LinearRegression()
        polynomial = PolynomialFeatures(degree=self.degree)
        self.coef = polynomial.fit_transform(self.x_src)
        poly_regr.fit(self.coef, self.y_src)
        self.y_polynomial = poly_regr.predict(polynomial.fit_transform(self.x_fit))

    def linear_regression(self):
        lin_regr = LinearRegression()
        lin_regr.fit(self.x_src, self.y_src)
        return lin_regr.predict(self.x_fit)

    @property
    def square_error(self):
        return mean_squared_error(self.y_src, self.y_polynomial)

    @property
    def root_score(self):
        return r2_score(self.y_src, self.y_polynomial)

    def plot(self):
        plt.scatter(self.x_src, self.y_src, label='Source points')
        plt.plot(self.x_fit, self.y_polynomial, label='Predict function', linestyle='--', color='orange')
        plt.legend(loc='upper left')
        plt.show()
