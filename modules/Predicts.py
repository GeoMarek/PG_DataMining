from typing import Tuple, Optional

import numpy as np
from matplotlib import pyplot as plt
from pandas._typing import ArrayLike
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


class RegressionPrediction:
    def __init__(self, x_src, y_src, degree=2):
        self.x_src = x_src
        self.y_src = y_src
        self.degree = degree
        self.x_fit = np.arange(self.x_src.min(), self.x_src.max()+1, 1)[:, np.newaxis]
        self.line_regr = LinearRegression()
        self.poly_regr = PolynomialFeatures(degree=self.degree)
        self.coef = self.poly_regr.fit_transform(self.x_src)
        self.line_regr.fit(self.coef, self.y_src)
        self.y_polynomial = self.line_regr.predict(self.poly_regr.fit_transform(self.x_fit))

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

    def predict_for_value(self, value: float) -> Optional[int]:
        count = 1
        first_src = self.x_src.min()
        last_src = self.x_src.max()
        src_max_value = self.y_src.max()
        while True:
            all_x = np.arange(first_src, last_src + count, 1)[:, np.newaxis]
            all_y = self.line_regr.predict(self.poly_regr.fit_transform(all_x))
            new_y = all_y.copy()[last_src:]
            if np.any(new_y < src_max_value):
                # print(f"Prediction with {self.degree} degree is bad (new values goes down)")
                return
            elif count > self.x_src.size:
                # print(f"Prediction with {self.degree} degree is bad (predicting date is to far)")
                return
            elif np.any(new_y >= value):
                # print(f"Prediction with {self.degree} degree is good. "
                #      f"It can be to achieve {value} value in {count} days for now")
                return count
            count += 1
