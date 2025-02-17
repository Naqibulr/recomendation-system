import ast
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class CollaborativeFilteringModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.interaction_matrix = None
        self.similarity_matrix = None

    def fit(self, method="cosine", top_k=100):
        """Creates the user-item interaction matrix and computes the similarity matrix."""
        self.interaction_matrix = self.create_interaction_matrix()

        if method == "cosine":
            print("Calculating cosine similarity...")
            self.similarity_matrix = self.calculate_sparse_cosine_similarity(top_k)
        else:
            raise ValueError("Invalid similarity method. Choose 'cosine'.")

    def predict(self, user_id, top_n=10):
        """
        Predicts the top N items for a given user using collaborative filtering.
        """
        if user_id not in self.interaction_matrix.index:
            print(f"User {user_id} not found in interaction matrix.")
            return pd.Series(dtype=float)

        # Get similarity scores
        user_index = self.interaction_matrix.index.get_loc(user_id)
        user_similarities = self.similarity_matrix[user_index].toarray().flatten()

        # Mask self-similarity and find the most similar users
        user_similarities[user_index] = 0  # Exclude the user themselves
        top_user_indices = np.argsort(user_similarities)[-10:][::-1]  # Top similar users

        # Compute weighted sum of similar users' interactions
        similar_users = self.interaction_matrix.iloc[top_user_indices]
        similarity_weights = user_similarities[top_user_indices]

        # Weighted item scores
        weighted_scores = (similar_users.T @ similarity_weights) / similarity_weights.sum()
        weighted_scores = weighted_scores.sort_values(ascending=False)

        # Remove items already interacted with
        rated_items = self.interaction_matrix.loc[user_id]
        recommendations = weighted_scores[~rated_items.index.isin(rated_items[rated_items > 0].index)]

        return recommendations.head(top_n)

    def calculate_sparse_cosine_similarity(self, top_k=100):
        """Computes a sparse cosine similarity matrix for users."""
        interaction_matrix_sparse = csr_matrix(self.interaction_matrix)

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(interaction_matrix_sparse, dense_output=False)

        # Keep only top-K similar users per row
        for i in range(similarity_matrix.shape[0]):
            row = similarity_matrix[i].toarray().flatten()
            top_k_indices = np.argpartition(row, -top_k)[-top_k:]  # More efficient than sorting everything
            mask = np.ones_like(row, dtype=bool)
            mask[top_k_indices] = False
            row[mask] = 0  # Zero out small similarities
            similarity_matrix[i] = csr_matrix(row)  # Convert back to sparse

        return similarity_matrix

    def create_interaction_matrix(self):
        """Creates a user-item interaction matrix from the MIND dataset."""
        # Ensure impressions column is correctly formatted
        self.df["impressions"] = self.df["impressions"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Convert list of tuples into DataFrame format
        interactions = self.df.explode("impressions")
        interactions[["news_id", "click"]] = pd.DataFrame(interactions["impressions"].tolist(), index=interactions.index)
        interactions = interactions[["user_id", "news_id", "click"]]

        # Create pivot table (user-item matrix)
        interaction_matrix = interactions.pivot_table(index="user_id", columns="news_id", values="click", fill_value=0)
        
        print(f"Interaction matrix created with shape: {interaction_matrix.shape}")
        return interaction_matrix
