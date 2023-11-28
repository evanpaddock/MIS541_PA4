from ast import Dict


def main():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from statsmodels.formula.api import ols

    df = pd.read_csv("pa4_reg_data.txt", sep="\t")
    # !Regression

    # !Step One
    def get_age(year: float):
        age = 2015 - year
        return age

    df["age"] = df["yr_built"].apply(get_age)

    # !Step Two
    print(
        df[
            ["sqft_living", "price", "grade", "floors", "age", "yr_built", "view"]
        ].describe(),
        end="\n\n",
    )

    # !Step Three
    ols_model = ols("price ~ sqft_living + grade + floors + age + view", df).fit()
    df["price_pred"] = ols_model.predict(df)
    print(ols_model.summary(), end="\n\n")

    # !Step Four
    plt.figure()
    plt.scatter(df["sqft_living"], df["price"], color="blue", alpha=0.4, s=3)
    plt.scatter(df["sqft_living"], df["price_pred"], color="orange", alpha=0.4, s=3)
    plt.title("Sqft Living vs. House Price")
    plt.xlabel("Sqft Living")
    plt.ylabel("House Price($ millions)")
    # plt.show()

    # !Step Five
    def get_mean_absolute_error(df: pd.DataFrame):
        MAE = 0
        for index, row in df.iterrows():
            MAE += abs(row["price"] - row["price_pred"])

        return MAE / len(df)

    def get_mean_sqrd_error(df: pd.DataFrame):
        MSE = 0
        for index, row in df.iterrows():
            MSE += (row["price"] - row["price_pred"]) ** 2

        return MSE / len(df)

    MAE = get_mean_absolute_error(df)
    MSE = get_mean_sqrd_error(df)
    RMSE = MSE**0.5

    print("Mean absolute error: " + str(MAE))
    print("Mean absolute error: " + str(RMSE))


# TODO: Natural Language Processing

main()
