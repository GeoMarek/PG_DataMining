from typing import List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame


def heat_map(_df: DataFrame, _cols: List[str]) -> None:
    cm = np.corrcoef(_df[_cols].values.T)
    sns.set(font_scale=1.5)
    _ = sns.heatmap(cm,
                    cbar=True,
                    annot=True,
                    square=True,
                    fmt='.2f',
                    annot_kws={'size': 15},
                    yticklabels=_cols,
                    xticklabels=_cols)
    plt.show()


def correlation_map(_df: DataFrame, _cols: List[str]) -> None:
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(_df[_cols], height=2.5)
    plt.show()


def lin_regplot(_x, _y, model):
    plt.scatter(_x, _y, c='blue')
    plt.plot(_x, model.predict(_x), color='red')
    return None
