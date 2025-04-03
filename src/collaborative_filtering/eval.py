import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from src.collaborative_filtering.new_model import MINDRecommender
from src.data_loader import MINDDataLoader


class CollaborativeFilteringEvaluator:
    def __init__(self, model, data_loader):
        self.model: MINDRecommender = model  # The collaborative filtering model
        self.data_loader: MINDDataLoader = data_loader  # The MINDDataLoader

    def evaluate(self, k=10):
        # Load the validation set
        df_behaviors_validation, _ = self.data_loader.load_interactions("validation")
        df_behaviors_validation = self.model.preprocessor.preprocess_interactions(df_behaviors_validation)

        # Get the common users between the validation set and the user-news matrix
        common_users = [user for user in df_behaviors_validation["user_id"].unique() if user in self.model.user_index]

        # Get the actual and predicted values for each user
        predicted = [
            self.model.recommend(user, top_k=k)
            for user in common_users
        ]

        # Get actual values, remember that the impressions are like this: ((N6400, 0), (N13353, 1), (N29862, 0),
        actual = [
            [impression[0] for impression in df_behaviors_validation[df_behaviors_validation["user_id"] == user]["impressions"].explode()]
            for user in common_users
        ]

        # Calculate evaluation metrics
        precision = precision_score(actual, predicted, average="micro")
        recall = recall_score(actual, predicted, average="micro")

        return {"precision": precision, "recall": recall}