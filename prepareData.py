import os
import pandas as pd
from Countries import Country


def readData():
    return pd.read_csv(os.path.join(
        os.getcwd(), 
        "data", 
        "data_12_05",
        "country_vaccinations.csv"
    ))


if __name__ == "__main__":
    global_df = readData()
    for country in global_df.country.unique():
        x = Country(global_df, country)
