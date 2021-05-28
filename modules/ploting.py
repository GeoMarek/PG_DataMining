import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def heat_map(_df, _cols):
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


def correlation_map(_df, _cols):
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(_df[_cols], height=2.5)
    plt.show()
