import numpy as np

from modules.Predicts import predict_with_regression, RegressionPrediction
from modules.files import read_data


def main():
    # save_clean_country("data_12_05", "Poland")
    df = read_data("Poland_data_12_05.csv")
    y = df['daily_vaccinations'].to_numpy().reshape(-1, 1)
    x = np.array(range(y.size)).reshape(-1, 1)
    # two_regressions(x.reshape(-1, 1),
    #                 y.reshape(-1, 1),
    #                 _degree=15)

    # x_fit, y_fit = predict_with_regression(x.reshape(-1, 1),
    #                                        y.reshape(-1, 1),
    #                                        degree=15)
    # plot_prediction(x, y, x_fit, y_fit)
    # square_error(y, y_fit)

    p = RegressionPrediction(x, y)
    # print(f"Prediction error: {p.square_error}")
    # p.plot()


if __name__ == "__main__":
    main()
