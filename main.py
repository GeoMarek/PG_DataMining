import os
from modules.common_functions import regression_from, init_country_directory, save_leaders

ARAB_EMIRATES = os.path.join(os.getcwd(), "data", "leaders", "pos002_United Arab Emirates.csv")
ISRAEL = os.path.join(os.getcwd(), "data", "leaders", "pos005_Israel.csv")
CHILE = os.path.join(os.getcwd(), "data", "leaders", "pos014_Chile.csv")
POLAND = os.path.join(os.getcwd(), "data", "countries", "Poland.csv")


def main():
    init_country_directory()
    save_leaders(25)
    print("United Arab Emirates")
    regression_from(ARAB_EMIRATES, 'people_fully_vaccinated_per_hundred', 80)
    print("Israel")
    regression_from(ISRAEL, 'people_fully_vaccinated_per_hundred', 80)
    print("Chile")
    regression_from(CHILE, 'people_fully_vaccinated_per_hundred', 80)
    print("Poland")
    regression_from(POLAND, 'people_fully_vaccinated_per_hundred', 80)


if __name__ == "__main__":
    main()
