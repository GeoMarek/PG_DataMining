from pandas import DataFrame
import pandas as pd
import os
from modules.Datasets import Dataset


def save_clean_country(directory: str, country: str) -> None:
    _file = Dataset(directory)
    _file.dataset = _file.dataset.loc[_file.dataset['country'] == country]
    _file.dataset.to_csv(f"{country}_{directory}.csv")


def save_clean(directory: str) -> None:
    _file = Dataset(directory)
    _file.dataset.to_csv(f"world_data_{directory}.csv")


def read_data(filename: str) -> DataFrame:
    return pd.read_csv(os.path.join(os.getcwd(), filename))
