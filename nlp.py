from turtle import color
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import re
import numpy as np
import calendar


def main():
    # //nlp_df = pd.read_csv("pa4_nlp_data.txt", sep="\t")
    nlp_df = pd.read_csv("nlp_with_sentiment.txt", sep="\t")

    # !Step one
    def get_month(dateString: str):
        MDY = dateString.split(" ")
        MDY[0] = re.sub(f"\D", "", MDY[0])

        if MDY[0][0] == "0":
            MDY[0] = MDY[0][1:]

        month = MDY[0]
        return int(month)

    def get_year(dateString: str):
        MDY = dateString.split(" ")
        year = MDY[2]
        return int(year)

    def get_word_count(comment: str):
        if comment == "" or pd.isna(comment):
            return 0
        comment = comment.lower()

        clean_comment = re.sub(r"[^a-zA-Z\s]", "", comment)
        clean_comment = re.sub(r"\s+", " ", clean_comment)
        comment_list = clean_comment.strip().split(" ")

        return len(comment_list)

    nlp_df["month"] = nlp_df["reviewTime"].apply(get_month)
    nlp_df["year"] = nlp_df["reviewTime"].apply(get_year)
    nlp_df["word_count"] = nlp_df["reviewText"].apply(get_word_count)

    # !Step 2
    def get_sentiment_score(comment: str):
        if type(comment) != str:
            return 0
        blob = TextBlob(comment)
        return blob.sentiment.polarity

    def get_sentiment_type(sentiment_score: float):
        if sentiment_score > 0:
            return "positive"
        elif sentiment_score < 0:
            return "negative"
        else:
            return "neutral"

    # //nlp_df["sentiment_score"] = nlp_df["reviewText"].apply(get_sentiment_score)
    # //nlp_df["sentiment_type"] = nlp_df["sentiment_score"].apply(get_sentiment_type)

    # !Step 3

    print(nlp_df[["overall", "word_count", "sentiment_score"]].describe(), end="\n\n")
    # //nlp_df.to_csv("nlp_with_sentiment.txt", sep="\t")

    # !Step 4
    reviews_per_month_df = nlp_df["month"].value_counts().to_frame()
    reviews_per_month_df = reviews_per_month_df.reset_index().sort_values(by="month")

    y_ticks = np.arange(0, 5500, 500)
    month_names = [
        calendar.month_name[month] for month in reviews_per_month_df["month"].values
    ]
    plt.figure()
    plt.bar(month_names, reviews_per_month_df["count"])
    plt.plot(reviews_per_month_df["month"], reviews_per_month_df["count"], color="red")
    plt.title("Number of Reviews by Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Reviews")
    plt.yticks(y_ticks)
    plt.xticks(rotation=45)
    plt.show()


main()
