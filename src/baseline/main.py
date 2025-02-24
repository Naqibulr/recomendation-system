import ast
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

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

    def predict(self, top_n=10):
        """Predicts the top N items for a given user using collaborative filtering."""
        top_items = self.popularityList.head(top_n).copy()
        top_items["score"] = top_items["click_count"] / top_items["click_count"].max()
        return top_items.set_index("news_id")["score"]
