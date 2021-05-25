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

    print(global_df['total_vaccinations'].isna().sum())
    print(global_df['total_vaccinations_per_hundred'].isna().sum())
    print(global_df['daily_vaccinations'].isna().sum())
    print(global_df['daily_vaccinations_per_million'].isna().sum())
    print(global_df['people_vaccinated'].isna().sum())
    print(global_df['people_vaccinated_per_hundred'].isna().sum())
    print(global_df['people_fully_vaccinated'].isna().sum())
    print(global_df['people_fully_vaccinated_per_hundred'].isna().sum())

    for country in global_df.country.unique():
        x = Country(global_df, country)
        global_df.loc[global_df['country'] == country] = x.df

    print(global_df['total_vaccinations'].isna().sum())
    print(global_df['total_vaccinations_per_hundred'].isna().sum())
    print(global_df['daily_vaccinations'].isna().sum())
    print(global_df['daily_vaccinations_per_million'].isna().sum())
    print(global_df['people_vaccinated'].isna().sum())
    print(global_df['people_vaccinated_per_hundred'].isna().sum())
    print(global_df['people_fully_vaccinated'].isna().sum())
    print(global_df['people_fully_vaccinated_per_hundred'].isna().sum())
