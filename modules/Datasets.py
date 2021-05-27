from typing import Iterator
import pandas as pd
import os
from pandas import DataFrame
from modules.Countries import Country


class Dataset:
    def __init__(self, filename):
        self.dataset = self.readDataFrom(filename)
        self._convertDate()
        unstandarized_countries = self.getDataWithWrongISO()
        for country in unstandarized_countries:
            self.dropDuplicatedRegion(country)
        self.fillEmptyData()

    def fillEmptyData(self):
        """
        Fill `NaN` values in dataset
        """
        for country in self.dataset.country.unique():
            sub_df = Country(self.dataset, country)
            self.dataset.loc[self.dataset['country'] == country] = sub_df.df
            print(f"{country} is done")

    @staticmethod
    def readDataFrom(file_date: str) -> DataFrame:
        """
        Read data from file. File is in directory with date of data acquisition in its name

        :param file_date: date of data acquisition
        :return: DataFrame
        """
        df = pd.read_csv(os.path.join(
            os.getcwd(),
            "data",
            file_date,
            "country_vaccinations.csv"
        ))
        return df

    def getDataWithWrongISO(self) -> Iterator[str]:
        """
        Returns a generator with subsets of data. Each subset refers to a different country with unstandarized iso

        :return: Iterator[DataFrame]
        """
        for code in self.dataset.iso_code.unique():
            if len(code) != 3:
                yield code

    def dropDuplicatedRegion(self, wrong_iso_code: str) -> None:
        """
        Drop rows with iso_code equal to argument
        :param wrong_iso_code:
        :return:
        """
        self.dataset = self.dataset.drop(self.dataset[self.dataset.iso_code == wrong_iso_code].index)

    def _convertDate(self):
        self.dataset['date'] = pd.to_datetime(self.dataset['date'], format='%Y-%m-%d')



