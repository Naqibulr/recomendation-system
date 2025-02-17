# Metrics for evaluation of the model


import numpy as np

def hit_rate_at_k(actual, predicted, k=10):
    """Compute Hit Rate @ K: Checks if at least one relevant item is in top-K recommendations."""
    hits = [1 if len(set(act) & set(pred[:k])) > 0 else 0 for act, pred in zip(actual, predicted)]
    return np.mean(hits)

def dcg_at_k(relevance, k):
    """Compute Discounted Cumulative Gain (DCG) @ K."""
    return np.sum([(rel / np.log2(i + 2)) for i, rel in enumerate(relevance[:k])])

def ndcg_at_k(actual, predicted, k=10):
    """Compute Normalized Discounted Cumulative Gain (NDCG) @ K."""
    def get_relevance(act, pred):
        return [1 if item in act else 0 for item in pred]
    
    scores = [dcg_at_k(get_relevance(act, pred), k) / (dcg_at_k(sorted(get_relevance(act, pred), reverse=True), k) + 1e-10)
              for act, pred in zip(actual, predicted)]
    return np.mean(scores)

def precision_at_k(actual, predicted, k=10):
    """Compute Precision @ K."""
    precisions = [len(set(act) & set(pred[:k])) / k for act, pred in zip(actual, predicted)]
    return np.mean(precisions)

def recall_at_k(actual, predicted, k=10):
    """Compute Recall @ K."""
    recalls = [len(set(act) & set(pred[:k])) / len(act) if len(act) > 0 else 0 for act, pred in zip(actual, predicted)]
    return np.mean(recalls)

def coverage(predicted, total_items):
    """Compute Coverage: The proportion of items recommended at least once."""
    unique_recommended = set(item for rec_list in predicted for item in rec_list)
    return len(unique_recommended) / total_items

def diversity(predicted, item_embeddings):
    """Compute Diversity: Measures how different recommended items are using cosine similarity."""
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    
    diversities = []
    for rec_list in predicted:
        pairwise_sims = [cosine_similarity(item_embeddings[i], item_embeddings[j])
                          for idx, i in enumerate(rec_list) for j in rec_list[idx+1:]]
        diversities.append(1 - np.mean(pairwise_sims) if pairwise_sims else 0)
    return np.mean(diversities)

def evaluate_model(model,test_df):
    """Evaluate the model on test data."""
    actual = [set(test_df[test_df["user_id"] == user]["impressions"].explode()) 
              for user in test_df["user_id"].unique()]
    
    predicted = [model.predict(user, top_n=k).index.tolist() 
                 if user in model.interaction_matrix.index else [] 
                 for user in test_df["user_id"].unique()]
    return {
        "HR@10": hit_rate_at_k(actual, predicted, k=10),
        "NDCG@10": ndcg_at_k(actual, predicted, k=10),
        "Precision@10": precision_at_k(actual, predicted, k=10),
        "Recall@10": recall_at_k(actual, predicted, k=10),
    }

# Example usage
if __name__ == "__main__":
    actual = [[1, 2, 3], [2, 5, 7], [10]]  # Ground truth (clicked articles per user)
    predicted = [[3, 1, 4, 5], [5, 2, 6, 7], [8, 9, 10, 11]]  # Recommended articles per user
    total_items = 15
    item_embeddings = {i: np.random.rand(5) for i in range(total_items)}  # Random 5D embeddings for items
    
    print("HR@10:", hit_rate_at_k(actual, predicted, k=10))
    print("NDCG@10:", ndcg_at_k(actual, predicted, k=10))
    print("Precision@10:", precision_at_k(actual, predicted, k=10))
    print("Recall@10:", recall_at_k(actual, predicted, k=10))
    print("Coverage:", coverage(predicted, total_items))
    print("Diversity:", diversity(predicted, item_embeddings))
