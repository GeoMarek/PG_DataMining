import math
import pandas as pd


class Country:

    def __init__(self, data_frame, country):
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

        # fill people vaccinated, not implemented
        self.fillPeopleVaccinated()
        self.fillPeopleVaccinatedPerHundred()

        # fill people fully vaccinated, not implemented
        self.fillPeopleFullyVaccinated()
        self.fillPeopleFullyVaccinatedPerHundred()

        # check for missings
        self.showMisses()

    def showMisses(self):
        x1 = self.df['total_vaccinations'].isna().sum()
        x2 = self.df['total_vaccinations_per_hundred'].isna().sum()
        x3 = self.df['daily_vaccinations'].isna().sum()
        x4 = self.df['daily_vaccinations_per_million'].isna().sum()
        x5 = self.df['people_vaccinated'].isna().sum()
        x6 = self.df['people_vaccinated_per_hundred'].isna().sum()
        x7 = self.df['people_fully_vaccinated'].isna().sum()
        x8 = self.df['people_fully_vaccinated_per_hundred'].isna().sum()

        if any([x1, x2, x3, x4, x5, x6, x7, x8]) > 0:
            print(f"Country {self.name.upper()} has missings")
            self._showMissingIndices(x1, 'total_vaccinations')
            self._showMissingIndices(x2, 'total_vaccinations_per_hundred')
            self._showMissingIndices(x3, 'daily_vaccinations')
            self._showMissingIndices(x4, 'daily_vaccinations_per_million')
            self._showMissingIndices(x5, 'people_vaccinated')
            self._showMissingIndices(x6, 'people_vaccinated_per_hundred')
            self._showMissingIndices(x7, 'people_fully_vaccinated')
            self._showMissingIndices(x8, 'people_fully_vaccinated_per_hundred')

    def fillTotalVaccinations(self):
        dfc = self.df.copy()
        dfc['total_vaccinations'] = dfc['total_vaccinations'].interpolate()
        self.df = dfc.copy()
        self._updateBeginningOf('total_vaccinations')

    def fillTotalVaccinationsPerHundred(self):
        self._fillPairedColumns('total_vaccinations', 'total_vaccinations_per_hundred')

    def fillDailyVaccinations(self):
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

    def fillDailyVaccinationsPerMillion(self):
        self._fillPairedColumns('daily_vaccinations', 'daily_vaccinations_per_million', coef=1_000_000)

    def fillPeopleVaccinated(self):
        # TODO: implement
        pass

    def fillPeopleVaccinatedPerHundred(self):
        self._fillPairedColumns('people_vaccinated', 'people_vaccinated_per_hundred')

    def fillPeopleFullyVaccinated(self):
        # TODO: implement
        pass

    def fillPeopleFullyVaccinatedPerHundred(self):
        self._fillPairedColumns('people_fully_vaccinated', 'people_fully_vaccinated_per_hundred')

    def _updateBeginningOf(self, column):
        first_null_index = None
        for index, row in self.df.iterrows():
            if first_null_index is None:
                first_null_index = index
                break
        self.df.at[first_null_index, column] = 0
        dfc = self.df.copy()
        dfc[column] = dfc[column].interpolate()
        self.df = dfc.copy()

    def _movingAverageForDailyVaccinations(self, value, index, start_index):
        count = 1
        vacc_sum = value
        while index - 1 < start_index:
            vacc_sum += self.df.at[index, 'daily_vaccinations']
            index -= 1
            count += 1
            if count == 7:
                break
        return vacc_sum / count

    def _dropUnnecesaryColumns(self):
        del self.df['source_name']
        del self.df['source_website']
        del self.df['vaccines']
        del self.df['daily_vaccinations_raw']

    def _getPopulationBasedOn(self, column):
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

    def _setPrettyAccuratePopulation(self):
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

    def _fillPairedColumns(self, total_col, rel_col, coef=100):
        for index, row in self.df.iterrows():
            if pd.isna(self.df.at[index, rel_col]):
                actual_total = self.df.at[index, total_col]
                relative_value = (actual_total / self.population) * coef
                self.df.at[index, rel_col] = relative_value

    @staticmethod
    def _showMissingIndices(x, name):
        if x > 0:
            print(f"\t{name} has {x} nan values")
            # print(np.where(self.df[name].isna())[0].tolist())
