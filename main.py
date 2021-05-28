import numpy as np

from modules.Predicts import RegressionPrediction
from modules.files import read_data, save_clean_country
from modules.ploting import lin_regplot
import matplotlib.pyplot as plt


def poly_regression(x, y):
    score = 0
    best_deegree = 1

    for num in range(1, 100):
        p = RegressionPrediction(x, y, degree=num)
        p_sc = p.root_score
        if p_sc > score:
            score = p_sc
            best_deegree = num
        print(f"Degree {num} was checked..")
    print(f"Best prediction with {best_deegree} degree - r2 score is {round(score, 10)}")
    p = RegressionPrediction(x, y, degree=best_deegree)
    p.plot()


def main():
    # save_clean_country("data_26_04", "Poland")
    df = read_data("Poland_data_26_04.csv")
    y = df['total_vaccinations'].to_numpy().reshape(-1, 1)
    x = np.array(range(y.size)).reshape(-1, 1)

    from sklearn.tree import DecisionTreeRegressor
    tree = DecisionTreeRegressor(max_depth=5)
    tree.fit(x, y)
    sort_idx = x.flatten().argsort()
    lin_regplot(x[sort_idx], y[sort_idx], tree)
    plt.show()
    poly_regression(x, y)


if __name__ == "__main__":
    main()
