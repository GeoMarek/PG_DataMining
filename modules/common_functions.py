import operator
from math import sqrt

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
import pandas as pd
import os
from modules.Datasets import Dataset
from typing import Dict, Tuple
from datetime import datetime

from modules.Predicts import RegressionPrediction
from modules.ploting import lin_regplot


def read_data(path_name: str) -> DataFrame:
    """
    Read csv file and return pandas data frame object

    :param path_name: path to file
    :return: pandas dataframe object
    """
    return pd.read_csv(path_name)


def read_from(country: str) -> DataFrame:
    """
    Read from csv representing specific country and return frame object

    :param country: country name
    :return: pandas dataframe object
    """
    return read_data(os.path.join(os.getcwd(), "data", "countries", country))


def init_country_directory() -> None:
    """
    If not exist create dorectory for country csv files. Then clean
    and fill data for each country. Save it in initiated directory.
    """
    path_name = os.path.join(os.getcwd(), "data", "countries")
    if not os.path.exists(path_name):
        os.makedirs(path_name)
        df = Dataset()
        df.fillAndSaveCountryData()


def vaccine_country_dict() -> Dict[str, int]:
    """
    Create and return a dict from country directory. Each pair in this
    dict contain max value from 'people_fully_vaccinated_per_hundred'
    and name of country.

    :return: { country_name : fully_vaccinated }
    """
    dir_name = os.path.join(os.getcwd(), "data", "countries")
    countries = {}
    for country in os.listdir(dir_name):
        df = read_data(os.path.join(dir_name, country))
        countries[country] = df['people_fully_vaccinated_per_hundred'].max()
    return countries


def get_vaccine_leaders(head: int, countries: Dict[str, int]) -> str:
    """
    Get list of country names which have the best vaccination program

    :param head: number of leaders from top
    :param countries: dictionary with people fully vaccinated in countries
    :return: country names
    """
    for _ in range(head):
        leader = max(countries.items(), key=operator.itemgetter(1))[0]
        del countries[leader]
        yield leader


def save_leader(position: int, name: str, data: DataFrame) -> None:
    """
    Save choosen country in leaders directory

    :param position: position in rank
    :param name: name of country
    :param data: country dataframe
    """
    path_name = os.path.join(os.getcwd(), "data", "leaders")
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    data.to_csv(os.path.join(path_name, f"pos{position:03d}_{name}"))


def save_leaders(head: int) -> None:
    """
    Save best countries in specific directory

    :param head: number of leaders
    """
    countries = vaccine_country_dict()
    count = 1
    for leader in get_vaccine_leaders(head, countries):
        df = read_from(leader)
        save_leader(count, leader, df)
        count += 1


def poly_regression_target(x, y, target, **kwargs) -> Tuple[float, int]:
    score = 0
    best_deegree = 1
    best_steps = None
    for num in range(1, 10):
        p = RegressionPrediction(x, y, degree=num)
        p_sc = p.root_score
        steps = p.predict_for_value(target)
        if p_sc > score and steps is not None:
            score = p_sc
            best_deegree = num
            best_steps = steps
    p = RegressionPrediction(x, y, degree=best_deegree)
    p.predict_for_value(target)
    p.plot(**kwargs)
    print(f"Final score: {score} ({best_deegree} degree) and achieve goal in {best_steps} days from now")
    return score, best_deegree


def poly_regression(x, y, **kwargs):
    score = 0
    best_deegree = 1
    for num in range(1, 5):
        p = RegressionPrediction(x, y, degree=num)
        p_sc = p.root_score
        if p_sc > score:
            score = p_sc
            best_deegree = num
    p = RegressionPrediction(x, y, degree=best_deegree)
    p.predict_future_values_in(15)
    p.plot(**kwargs)
    print(f"Final score: {score} ({best_deegree} degree)")
    return p.new_y[p.x_src.max():]


def calculate_diff(real_list, pred_list):
    diffs = []
    for real, pred in zip(real_list, pred_list):
        difference = pred-real
        print(f"{real[0]} <-> {int(difference[0])} <-> {round(100*difference[0]/real[0],2)}")
        diffs.append(abs(difference))
    return sqrt(sum([i**2 for i in diffs])/(len(diffs)-2))


def random_tree_regression(x, y):
    from sklearn.tree import DecisionTreeRegressor
    tree = DecisionTreeRegressor(max_depth=5)
    tree.fit(x, y)
    sort_idx = x.flatten().argsort()
    lin_regplot(x[sort_idx], y[sort_idx], tree)
    plt.show()


def regression_from(filename: str, column_name: str, target: int, **kwargs) -> Tuple[float, int]:
    df = read_data(filename)
    y = df[column_name].to_numpy().reshape(-1, 1)
    x = np.array(range(y.size)).reshape(-1, 1)
    return poly_regression_target(x, y, target, **kwargs)


def predict_vaccine_demand(dataset, column):
    df = read_data(dataset)
    y = df[column].to_numpy().reshape(-1, 1)
    x = np.array(range(y.size)).reshape(-1, 1)
    return poly_regression(x, y, begin="2020-12-28", title="Predykcja ilości podanych dawek do 15 maja")


def prepare_files_to_predict_demand(base_file: str) -> Tuple[str, str]:
    test_path = os.path.join(os.getcwd(), "data", "testing_set.csv")
    learn_path = os.path.join(os.getcwd(), "data", "learning_set.csv")
    pick_rows_to("2021-04-30", base_file, "learning_set")
    pick_rows_to("2021-05-15", base_file, "testing_set")
    pick_rows_from("2021-05-01", test_path)
    return test_path, learn_path


def pick_rows_to(end_date: str, filename: str, save_name: str) -> None:
    df = read_data(filename)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= datetime.strptime(end_date, '%Y-%m-%d')]
    df.to_csv(os.path.join(os.getcwd(), "data", f"{save_name}.csv"))


def pick_rows_from(start_date: str, filename: str) -> None:
    df = read_data(filename)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= datetime.strptime(start_date, '%Y-%m-%d')]
    df.to_csv(filename)
