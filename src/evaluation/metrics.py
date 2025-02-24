# Metrics for evaluation of the model


import numpy as np


class EvaluationMetrics:
    """Class to compute evaluation metrics for both cf and cb models."""

    def __init__(self, actual, predicted):
        self.actual = actual
        self.predicted = predicted

    def hit_rate_at_k(self, k=10):
        """Compute Hit Rate @ K: Checks if at least one relevant item is in top-K recommendations."""
        hit_rates = []
        for act, pred in zip(self.actual, self.predicted):
            hit_rates.append(len(set(act) & set(pred[:k])) > 0)
        return np.mean(hit_rates)

    def dcg_at_k(self, relevance, k):
        """Compute Discounted Cumulative Gain (DCG) @ K."""
        dcg = 0
        for i, rel in enumerate(relevance[:k]):
            dcg += rel / np.log2(i + 2)
        return dcg

    def ndcg_at_k(self, k=10):
        """Compute Normalized Discounted Cumulative Gain (NDCG) @ K."""
        def get_relevance(act, pred):
            return [1 if item in act else 0 for item in pred]

        scores = []
        for act, pred in zip(self.actual, self.predicted):
            dcg = self.dcg_at_k(get_relevance(act, pred), k)
            idcg = self.dcg_at_k(
                sorted(get_relevance(act, pred), reverse=True), k)
            scores.append(dcg / (idcg + 1e-10))
        return np.mean(scores)

    def precision_at_k(self, k=10):
        """Compute Precision @ K."""
        precisions = []
        for act, pred in zip(self.actual, self.predicted):
            precisions.append(len(set(act) & set(pred[:k])) / k)
        return np.mean(precisions)

    def recall_at_k(self, k=10):
        """Compute Recall @ K."""
        recalls = []
        for act, pred in zip(self.actual, self.predicted):
            recalls.append(
                len(set(act) & set(pred[:k])) / len(act) if len(act) > 0 else 0)
        return np.mean(recalls)

    def coverage(self, total_items):
        """Compute Coverage: The proportion of items recommended at least once."""
        unique_recommended = set(
            item for rec_list in self.predicted for item in rec_list)
        return len(unique_recommended) / total_items

    def diversity(self, item_embeddings):
        """Compute Diversity: Measures how different recommended items are using cosine similarity."""
        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

        diversities = []
        for rec_list in self.predicted:
            pairwise_sims = [cosine_similarity(item_embeddings[i], item_embeddings[j])
                             for idx, i in enumerate(rec_list) for j in rec_list[idx+1:]]
            diversities.append(1 - np.mean(pairwise_sims)
                               if pairwise_sims else 0)
        return np.mean(diversities)

    def evaluate_model(self, model, test_df, k=10):
        """Evaluate the model on test data."""
        self.actual = [set(test_df[test_df["user_id"] == user]["impressions"].explode())
                       for user in test_df["user_id"].unique()]

        self.predicted = [model.predict(user, top_n=k).index.tolist()
                          if user in model.interaction_matrix.index else []
                          for user in test_df["user_id"].unique()]

        return {
            "HR@10": self.hit_rate_at_k(k=100),
            "NDCG@10": self.ndcg_at_k(k=100),
            "Precision@10": self.precision_at_k(k=100),
            "Recall@10": self.recall_at_k(k=100),
        }
