import os
from modules.common_functions import regression_from, init_country_directory, save_leaders, pick_rows_to, pick_rows_from

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
    pick_rows_to("2021-04-30", POLAND, "learning_set")
    pick_rows_to("2021-05-15", POLAND, "testing_set")
    pick_rows_from("2021-05-01", os.path.join(os.getcwd(), "data", "testing_set.csv"))


if __name__ == "__main__":
    init_country_directory()
    # predict_population_resistance()
    predict_vaccines_demand()
