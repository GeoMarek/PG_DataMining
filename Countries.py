import math
import pandas as pd
from pandas import DataFrame


class Country:

    def __init__(self, data_frame: DataFrame, country: str):
        self.df = data_frame.loc[data_frame['country'] == country]
        self.name = country
        self.population = 0

        # prepare dataset
        self._setPrettyAccuratePopulation()
        self._dropUnnecesaryColumns()

        # fill total vaccinations
        self.fillTotalVaccinations()
        self.fillTotalVaccinationsPerHundred()

        # fill daily vaccinations
        self.fillDailyVaccinations()
        self.fillDailyVaccinationsPerMillion()

        # fill people vaccinated
        self.fillPeopleVaccinated()
        self.fillPeopleVaccinatedPerHundred()
        self.fillPeopleFullyVaccinatedPerHundred()

    def fillTotalVaccinations(self) -> None:
        self._interpolateBy('total_vaccinations')
        self._updateBeginningOf('total_vaccinations')
        self.df['total_vaccinations'] = self.df['total_vaccinations'].astype(int)

    def fillTotalVaccinationsPerHundred(self) -> None:
        self._fillPairedColumns('total_vaccinations', 'total_vaccinations_per_hundred')

    def fillDailyVaccinations(self) -> None:
        start_id = None
        column = 'daily_vaccinations'
        other_column = 'total_vaccinations'
        for index, row in self.df.iterrows():
            if start_id is None:
                start_id = index
            if pd.isna(self.df.at[index, column]):
                if index == start_id:
                    value = 0
                else:
                    x1 = self.df.at[index-1, other_column]
                    x2 = self.df.at[index, other_column]
                    value = self._movingAverageForDailyVaccinations(x2 - x1, index, start_id)
                self.df.at[index, column] = value
        self._updateBeginningOf('daily_vaccinations')
        self.df['daily_vaccinations'] = self.df['daily_vaccinations'].astype(int)

    def fillDailyVaccinationsPerMillion(self) -> None:
        self._fillPairedColumns('daily_vaccinations', 'daily_vaccinations_per_million', coef=1_000_000)

    def fillPeopleVaccinated(self) -> None:
        self._fillRowsWithPeopleVaccinated()
        self._interpolateBy('people_vaccinated')
        self._updateBeginningOf('people_vaccinated')
        self._fillRowsWithPeopleVaccinated()
        self.df['people_vaccinated'] = self.df['people_vaccinated'].astype(int)
        self.df['people_vaccinated_per_hundred'] = self.df['people_vaccinated_per_hundred'].astype(int)

    def fillPeopleVaccinatedPerHundred(self) -> None:
        self._fillPairedColumns('people_vaccinated', 'people_vaccinated_per_hundred')

    def fillPeopleFullyVaccinatedPerHundred(self) -> None:
        self._fillPairedColumns('people_fully_vaccinated', 'people_fully_vaccinated_per_hundred')

    def _fillRowsWithPeopleVaccinated(self) -> None:
        """
        Fill in the columns that are dependent on each other:
        `peopleWithTwo = allPeople - peopleWithOne`
        `peopleWithOne = allPeople - peopleWithTwo`
        """
        for index, row in self.df.iterrows():
            total = self.df.at[index, 'total_vaccinations']
            people = self.df.at[index, 'people_vaccinated']
            fully = self.df.at[index, 'people_fully_vaccinated']
            is_people_nan = math.isnan(people)
            is_fully_nan = math.isnan(fully)
            if people > total:
                self.df.at[index, 'total_vaccinations'] = people
                if is_fully_nan:
                    self.df.at[index, 'people_fully_vaccinated'] = 0
            elif (not is_people_nan) and is_fully_nan:
                fully = total - people
                self.df.at[index, 'people_fully_vaccinated'] = fully
            elif is_people_nan and (not is_fully_nan):
                people = total - fully
                self.df.at[index, 'people_vaccinated'] = people

    def _interpolateBy(self, column_name: str) -> None:
        """
        Update data frame in column using default interpolation method

        :param column_name: name of column
        """
        dfc = self.df.copy()
        dfc[column_name] = dfc[column_name].interpolate()
        self.df = dfc.copy()

    def _updateBeginningOf(self, column_name: str) -> None:
        """
        Update data frame in specific column if at the beginning is NaN value.
        Fill this NaN value with 0 and then interpolate values in column.

        :param column_name: name of column

        """
        first_null_index = None
        for index, row in self.df.iterrows():
            if first_null_index is None:
                first_null_index = index
                break
        self.df.at[first_null_index, column_name] = 0

        self._interpolateBy(column_name)

    def _movingAverageForDailyVaccinations(self, value: int, index: int, start_index: int) -> int:
        """
        Calculate moving average using 6 values from behind and one from diff

        :param value: difference between total vaccinations
        :param index: numer of row in which will be new value
        :param start_index: index at first row
        :return: moving average
        """
        count = 1
        vacc_sum = value
        while index - 1 < start_index:
            vacc_sum += self.df.at[index, 'daily_vaccinations']
            index -= 1
            count += 1
            if count == 7:
                break
        return vacc_sum // count

    def _dropUnnecesaryColumns(self) -> None:
        """
        Drop useless columns
        """
        del self.df['source_name']
        del self.df['source_website']
        del self.df['vaccines']
        del self.df['daily_vaccinations_raw']

    def _getPopulationBasedOn(self, column: str) -> int:
        """
        Returns population based on equation:
         `population = (total / per_hundred) * 100`

        :param column: column on which calculations are based
        :return: population
        """
        abs_col = column
        rel_col = column + "_per_hundred"
        absolute = None
        div_sum = []
        for index, row in self.df.iterrows():
            relative = row[rel_col]
            absolute = row[abs_col]
            if not pd.isna(self.df.at[index, rel_col]) and relative != 0:
                div_sum.append(absolute / relative)
        if len(div_sum) < 1:
            return absolute
        else:
            return int(sum(div_sum) / len(div_sum)) * 100

    def _setPrettyAccuratePopulation(self) -> None:
        """
        Set population based on three columns. If some of them is NaN, don't include it
        """
        avg = [self._getPopulationBasedOn('total_vaccinations'),
               self._getPopulationBasedOn('people_vaccinated'),
               self._getPopulationBasedOn('people_fully_vaccinated')
               ]
        avg_sum = 0
        avg_size = 0
        for i in avg:
            if not math.isnan(i):
                avg_sum += i
                avg_size += 1
        self.population = avg_sum // avg_size

    def _fillPairedColumns(self, total_col: str, rel_col: str, coef: int = 100) -> None:
        """
        Update data frame relatice column basing on population:

        `x_per_hundred = (x / population) * 100

        :param total_col: column with total values
        :param rel_col: column with relative values
        :param coef: per hundred or per ...
        """
        for index, row in self.df.iterrows():
            if pd.isna(self.df.at[index, rel_col]):
                actual_total = self.df.at[index, total_col]
                relative_value = (actual_total / self.population) * coef
                self.df.at[index, rel_col] = relative_value
