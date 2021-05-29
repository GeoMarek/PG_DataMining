from modules.Predicts import RegressionPrediction
from modules.common_functions import init_country_directory, vaccine_country_dict, get_vaccine_leaders, read_from, \
    save_leader, save_leaders
from modules.ploting import lin_regplot
import matplotlib.pyplot as plt
import numpy as np


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


def random_tree_regression(x, y):
    from sklearn.tree import DecisionTreeRegressor
    tree = DecisionTreeRegressor(max_depth=5)
    tree.fit(x, y)
    sort_idx = x.flatten().argsort()
    lin_regplot(x[sort_idx], y[sort_idx], tree)
    plt.show()


def main():
    init_country_directory()
    save_leaders(25)




    # save_clean_country("data_26_04", "Israel")
    # df = read_data("Israel_data_26_04.csv")
    # y = df['total_vaccinations'].to_numpy().reshape(-1, 1)
    # x = np.array(range(y.size)).reshape(-1, 1)
    # TODO: dataset for 3 countries with biggest fully vaccinated + Poland
    # TODO: models for each country
    # TODO: predict for 70 percent
    # models for each country + Poland
    # TODO: Poland dataset to 31.04 (pl_learn)
    # TODO: Poland dataset to 15.05 (pl_test)
    # TODO: model from pl_learn
    # TODO: diff model and pl_test
    

if __name__ == "__main__":
    main()
