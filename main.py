import os
from modules.common_functions import regression_from, init_country_directory, save_leaders

LEADER1 = os.path.join(os.getcwd(), "data", "leaders", "pos002_United Arab Emirates.csv")
LEADER2 = os.path.join(os.getcwd(), "data", "leaders", "pos005_Israel.csv")
LEADER3 = os.path.join(os.getcwd(), "data", "leaders", "pos014_Chile.csv")
POLAND = os.path.join(os.getcwd(), "data", "countries", "Poland.csv")


def main():
    init_country_directory()
    save_leaders(25)
    print("United Arab Emirates")
    regression_from(LEADER1, 'people_fully_vaccinated_per_hundred')
    print("Israel")
    regression_from(LEADER2, 'people_fully_vaccinated_per_hundred')
    print("Chile")
    regression_from(LEADER3, 'people_fully_vaccinated_per_hundred')

    # TODO: znaleźć sposób na wybranie stopnia, gdzie predykcja idzie w górę i nie ma overfitting
    # TODO: jeśli jakiś predykowany y jest mniejszy niż maxY z y_src to odrzuć
    # TODO: zmniejszyć stopnie
    # TODO: metoda na stop kiedy procent odpornych wynosi 90
    # TODO: ładny wykres dla krajów wiodących
    # TODO: wycięcie z Polski okresu do 30 kwietnia (learn)
    # TODO: wycięcie z Polski okresu do 15 maja (test)
    # TODO: predykcja na learn ile będzie za 15 dni
    # TODO: obliczenie roznicy learn - test
    # TODO: ładny wykres dla Polski


if __name__ == "__main__":
    main()
