import os
import pandas as pd
from pandas import DataFrame

from Countries import Country


def readData() -> DataFrame:
    df = pd.read_csv(os.path.join(
        os.getcwd(),
        "data",
        "data_12_05",
        "country_vaccinations.csv"
    ))
    del df['source_name']
    del df['source_website']
    del df['vaccines']
    del df['daily_vaccinations_raw']
    return df


def fillDataFrame(data: DataFrame) -> None:
    for country in data.country.unique():
        sub_df = Country(data, country)
        data.loc[data['country'] == country] = sub_df.df
        print(f"{country} is done")