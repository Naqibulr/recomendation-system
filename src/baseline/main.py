import ast
import pandas as pd
import numpy as np

class PopularityModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.popularityList = None

    def fit(self):
        # Ensure impressions column is correctly formatted
        self.df["impressions"] = self.df["impressions"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Convert list of tuples into DataFrame format
        interactions = self.df.explode("impressions")

        interactions = pd.DataFrame(interactions["impressions"].tolist(), columns=["news_id", "click"], index=interactions.index)
        interactions = interactions[interactions["click"] == 1]
        popularityList = (interactions[interactions["click"] == 1]
                               .groupby("news_id").size()
                               .reset_index(name="click_count")
                               .sort_values("click_count", ascending=False))
        popularityList["score"] = popularityList["click_count"] - popularityList["click_count"].min()
        popularityList["score"] = popularityList["score"] / popularityList["score"].max()
        self.popularityList = popularityList

    def predict(self, items=None, top_n=10):
        """Predicts the top N items/supplied items for a given user using popularity scores."""
        if items is None:
            top_items = self.popularityList.head(top_n).copy()

            top_items["score"] = np.where(
                top_items["score"] > .01, 1, 0
            )
        else:
            top_items = self.popularityList.copy()
            top_items = top_items[top_items["news_id"].isin(items)]

        return top_items.set_index("news_id")["score"]

    def quantile(self):
        return self.popularityList["score"].quantile(0.75)

class RecencyModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.recencyList = None

    def fit(self):
        """Tracks the first recent impression timestamp for each news article."""
        # Ensure impressions column is correctly formatted
        self.df["impressions"] = self.df["impressions"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Convert list of tuples into DataFrame format
        interactions = self.df.explode("impressions")

        # Extract news_id values
        interactions = pd.DataFrame(interactions["impressions"].tolist(), columns=["news_id", "click"], index=interactions.index)
        interactions["timestamp"] = self.df.loc[interactions.index, "time"].values

        # Compute the earliest timestamp for each news article
        latest_timestamps = interactions.groupby("news_id")["timestamp"].min().reset_index()

        # Normalize the recency score (inverse of time difference)
        min_timestamp = latest_timestamps["timestamp"].min()
        max_timestamp = latest_timestamps["timestamp"].max()

        latest_timestamps["score"] = latest_timestamps["timestamp"].apply(lambda x: (x - min_timestamp) / (max_timestamp - min_timestamp))
        latest_timestamps = latest_timestamps.sort_values("timestamp", ascending=False).reset_index(drop=True)
        self.recencyList = latest_timestamps

    def predict(self, items=None, top_n=10):
        """Predicts the top N items/supplied items for a given user using recency scores."""
        if items is None:
            top_items = self.recencyList.head(top_n).copy()
        else:
            top_items = self.recencyList.copy()
            top_items = top_items[top_items["news_id"].isin(items)]

        return top_items.set_index("news_id")["score"]

    def quantile(self):
        return self.recencyList["score"].quantile(0.75)
