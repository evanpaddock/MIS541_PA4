from datetime import datetime
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import re
import numpy as np
import calendar


def main():
    nlp_df = pd.read_csv("pa4_nlp_data.txt", sep="\t")
    #  //nlp_df = pd.read_csv("nlp_with_sentiment.txt", sep="\t")

    # !Step one
    def get_month(dateString: str):
        date = datetime.strptime(dateString, "%m %d, %Y")
        month = date.month
        return month

    def get_year(dateString: str):
        date = datetime.strptime(dateString, "%m %d, %Y")
        year = date.year
        return year

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

    nlp_df["sentiment_score"] = nlp_df["reviewText"].apply(get_sentiment_score)
    nlp_df["sentiment_type"] = nlp_df["sentiment_score"].apply(get_sentiment_type)

    # !Step 3

    print(nlp_df[["overall", "word_count", "sentiment_score"]].describe(), end="\n\n")
    nlp_df.to_csv("nlp_with_sentiment.txt", sep="\t")

    # !Step 4
    reviews_per_month_df = nlp_df["month"].value_counts().to_frame()
    reviews_per_month_df = reviews_per_month_df.reset_index().sort_values(by="month")

    y_ticks = np.arange(0, 5500, 500)

    month_names = [
        calendar.month_name[month] for month in reviews_per_month_df["month"].values
    ]

    print(reviews_per_month_df, end="\n\n")

    plt.figure()
    plt.bar(month_names, reviews_per_month_df["count"])
    plt.plot(month_names, reviews_per_month_df["count"], color="red")
    plt.title("Number of Reviews by Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Reviews")
    plt.yticks(y_ticks)
    plt.xticks(rotation=45)
    # plt.show()

    # !Step 5
    review_sentiment_df = nlp_df.groupby(["overall"])["sentiment_type"].value_counts()
    review_sentiment_df = review_sentiment_df.reset_index()

    pivot_df = review_sentiment_df.pivot(
        index="overall", columns="sentiment_type", values="count"
    )

    y_ticks = np.arange(0, 30_000, 2_000)

    print(pivot_df, end="\n\n")

    pivot_df.plot(kind="bar", legend=True)
    plt.xlabel("Rating")
    plt.ylabel("Number of Ratings")
    plt.title("Sentiment Type Distribution Across Ratings")
    plt.yticks(y_ticks)
    # plt.show()

    # !Step 6
    average_sent_rating_df = (
        nlp_df.groupby("overall")["sentiment_score"].mean()
    ).to_frame()

    y_ticks = np.arange(0.000, 0.275, 0.025)

    print(average_sent_rating_df.reset_index(), end="\n\n")

    average_sent_rating_df.plot(kind="bar", legend=False)
    plt.xlabel("Rating")
    plt.ylabel("Average Sentiment Score")
    plt.yticks(y_ticks)
    plt.title("Average Sentiment Score Acress Ratings")

    # !Extras
    import seaborn as sns

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(
        nlp_df[nlp_df["word_count"] < 800]["word_count"], bins=100
    )  # Removing word_count > 800 to show marjority in more detail

    plt.xlabel("Count")
    plt.ylabel("Frequency")
    plt.title("Histogram of Word Count Distributions")

    overall_word_count_df = (
        nlp_df.groupby(["overall"])["word_count"].mean().to_frame().reset_index()
    )

    print(overall_word_count_df, end="\n\n")

    plt.figure()
    sns.barplot(
        x=overall_word_count_df["overall"], y=overall_word_count_df["word_count"]
    )
    plt.xlabel("Rating")
    plt.ylabel("Average Word Count")
    plt.title("Average Word Count by Rating")

    plt.show()


main()
