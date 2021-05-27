import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def heat_map(_df, _cols):
    cm = np.corrcoef(_df[_cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 15},
                     yticklabels=_cols,
                     xticklabels=_cols)
    plt.show()


def correlation_map(_df, _cols):
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(_df[_cols], height=2.5)
    plt.show()


def two_regressions(x_src, y_src):
    lr = LinearRegression()
    pr = LinearRegression()
    quadratic = PolynomialFeatures(degree=2)
    x_quad = quadratic.fit_transform(x_src)
    lr.fit(x_src, y_src)
    x_fit = np.arange(150, 175, 200)[:, np.newaxis]
    y_lin_fit = lr.predict(x_fit)
    pr.fit(x_quad, y_src)
    y_quad_fit = pr.predict(quadratic.fit_transform(x_fit))
    # plt.scatter(x_src, y_src, label='Punkty uczace')
    plt.plot(x_fit, y_lin_fit, label='Dopasowanie liniowe', linestyle='--')
    plt.plot(x_fit, y_quad_fit, label='Dopasowanie kwadratowe')
    plt.legend(loc='upper left')
    plt.show()
