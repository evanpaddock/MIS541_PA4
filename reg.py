from unicodedata import numeric
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    reg_df = pd.read_csv("pa4_reg_data.txt", sep="\t")
    # !Regression

    # !Step One
    def get_age(year: float):
        age = 2015 - year
        return age

    reg_df["age"] = reg_df["yr_built"].apply(get_age)

    # !Step Two
    print(
        reg_df[
            ["sqft_living", "price", "grade", "floors", "age", "yr_built", "view"]
        ].describe(),
        end="\n\n",
    )

    # !Step Three
    ols_model = ols("price ~ sqft_living + grade + floors + age + view", reg_df).fit()
    reg_df["price_pred"] = ols_model.predict(reg_df)
    print(ols_model.summary(), end="\n\n")

    # !Step Four
    plt.figure()
    plt.scatter(reg_df["sqft_living"], reg_df["price"], color="blue", alpha=0.4, s=3)
    plt.scatter(
        reg_df["sqft_living"], reg_df["price_pred"], color="orange", alpha=0.4, s=3
    )
    plt.title("Sqft Living vs. House Price")
    plt.xlabel("Sqft Living")
    plt.ylabel("House Price($ millions)")

    # !Step Five
    def get_mean_absolute_error(reg_df: pd.DataFrame):
        MAE = 0
        for index, row in reg_df.iterrows():
            MAE += abs(row["price"] - row["price_pred"])

        return MAE / len(reg_df)

    def get_mean_sqrd_error(reg_df: pd.DataFrame):
        MSE = 0
        for index, row in reg_df.iterrows():
            MSE += (row["price"] - row["price_pred"]) ** 2

        return MSE / len(reg_df)

    my_MAE = get_mean_absolute_error(reg_df)
    my_MSE = get_mean_sqrd_error(reg_df)
    my_RMSE = my_MSE**0.5

    MAE = mean_absolute_error(reg_df["price"], reg_df["price_pred"])
    MSE = mean_squared_error(reg_df["price"], reg_df["price_pred"])
    RMSE = MSE**0.5

    print("My mean absolute error: " + str(my_MAE))
    print("My root mean squared error: " + str(my_RMSE), end="\n\n")

    print("Mean absolute error: " + str(MAE))
    print("Root mean squared error: " + str(RMSE), end="\n\n")

    # !Extras
    import seaborn as sns

    # Checking for mulitcollinearity in data
    plt.figure(figsize=(14, 6))
    heatmap = sns.heatmap(
        reg_df.corr(numeric_only=True), cmap="coolwarm", annot=True, fmt=".2f"
    )
    heatmap.set_title("Correlation Heatmap", fontdict={"fontsize": 15}, pad=12)
    plt.xticks(rotation=45)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(
        reg_df["age"], bins=20, palette="viridis"
    )

    plt.xlabel("Age (Year Built - 2015)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Home Age Distribution")

    plt.show()


main()
