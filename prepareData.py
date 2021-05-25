from functions import fillDataFrame, readData


def main() -> None:
    df = readData()
    fillDataFrame(df)
    df.to_csv("clean_data.csv")


if __name__ == "__main__":
    main()
