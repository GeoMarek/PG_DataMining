import os
import numpy as np

from modules.common_functions import regression_from, save_leaders, prepare_files_to_predict_demand, read_data, \
    poly_regression, predict_vaccine_demand, calculate_diff
from modules.ploting import diff_plot

ARAB_EMIRATES = os.path.join(os.getcwd(), "data", "leaders", "pos002_United Arab Emirates.csv")
ISRAEL = os.path.join(os.getcwd(), "data", "leaders", "pos005_Israel.csv")
CHILE = os.path.join(os.getcwd(), "data", "leaders", "pos014_Chile.csv")
POLAND = os.path.join(os.getcwd(), "data", "countries", "Poland.csv")


def predict_population_resistance():
    save_leaders(25)
    regression_from(ARAB_EMIRATES,
                    'people_fully_vaccinated_per_hundred',
                    80,
                    title="Predykcja zaszczepienia populacji w 80% (Zjednoczone Emiraty Arabskie)",
                    ylabel="Zaszczepienie ludzie [%]",
                    begin="2021-01-05")
    regression_from(ISRAEL,
                    'people_fully_vaccinated_per_hundred',
                    80,
                    title="Predykcja zaszczepienia populacji w 80% (Izrael)",
                    ylabel="Zaszczepienie ludzie [%]",
                    begin="2020-12-19")
    regression_from(CHILE,
                    'people_fully_vaccinated_per_hundred',
                    80,
                    title="Predykcja zaszczepienia populacji w 80% (Chile)",
                    ylabel="Zaszczepienie ludzie [%]",
                    begin="2020-12-24")
    regression_from(POLAND,
                    'people_fully_vaccinated_per_hundred',
                    80,
                    title="Predykcja zaszczepienia populacji w 80% dla Polski",
                    ylabel="Zaszczepienie ludzie [%]",
                    begin="2020-12-28")


def predict_vaccines_demand():
    test, learn = prepare_files_to_predict_demand(POLAND)
    new_y = predict_vaccine_demand(learn, 'daily_vaccinations')
    y = read_data(test)['daily_vaccinations'].to_numpy().reshape(-1, 1)
    x = np.array(range(y.size)).reshape(-1, 1)
    diff_plot(x, new_y, y)
    print(calculate_diff(y, new_y))





if __name__ == "__main__":
    # init_country_directory()
    # predict_population_resistance()
    predict_vaccines_demand()
