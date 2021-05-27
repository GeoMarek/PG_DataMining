import numpy as np

from modules.files import read_data
from modules.ploting import two_regressions


def main():
    # save_clean_country("data_12_05", "Poland")
    df = read_data("Poland_data_12_05.csv")
    y = df['daily_vaccinations'].to_numpy()
    x = np.array(range(y.size))
    two_regressions(x.reshape(-1, 1), y.reshape(-1, 1))


if __name__ == "__main__":
    main()



