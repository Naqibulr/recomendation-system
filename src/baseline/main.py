import ast
import pandas as pd

class PopularityModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.popularityList = None

    def fit(self):
        """Creates a user-item interaction matrix from the MIND dataset."""
        # Ensure impressions column is correctly formatted
        self.df["impressions"] = self.df["impressions"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Convert list of tuples into DataFrame format
        interactions = self.df.explode("impressions")

        interactions = pd.DataFrame(interactions["impressions"].tolist(), columns=["news_id", "click"], index=interactions.index)
        interactions = interactions[interactions["click"] == 1]
        self.popularityList = (interactions[interactions["click"] == 1]
                               .groupby("news_id").size()
                               .reset_index(name="click_count")
                               .sort_values("click_count", ascending=False))

    def predict(self, items=None, top_n=10):
        """Predicts the top N items/supplied items for a given user using popularity scores."""
        if items is None:
            top_items = self.popularityList.head(top_n).copy()
        else:
            top_items = self.popularityList.copy()
            top_items = top_items[top_items["news_id"].isin(items)]

        top_items["score"] = top_items["click_count"] - top_items["click_count"].min()
        top_items["score"] = top_items["score"] / top_items["score"].max()
        return top_items.set_index("news_id")["score"]


class RecencyModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.latest_timestamps = None

    def fit(self):
        """Tracks the first recent impression timestamp for each news article."""
        # Ensure impressions column is correctly formatted
        self.df["impressions"] = self.df["impressions"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Convert list of tuples into DataFrame format
        interactions = self.df.explode("impressions")

        # Extract news_id values
        interactions = pd.DataFrame(interactions["impressions"].tolist(), columns=["news_id", "click"], index=interactions.index)
        interactions["timestamp"] = self.df.loc[interactions.index, "timestamp"].values

        # Compute latest timestamp for each news article
        self.latest_timestamps = interactions.groupby("news_id")["timestamp"].min().reset_index()

    def predict(self, top_n=10):
        """Returns the most recent N articles."""
        return self.latest_timestamps.sort_values("timestamp", ascending=False).head(top_n).set_index("news_id")
