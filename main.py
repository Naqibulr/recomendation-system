import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

from src.baseline.main import PopularityModel, RecencyModel
from src.baseline.preprocessing import BaselinePreprocessor
from src.data_loader import MINDDataLoader


# --- Ranking Metrics ---
def dcg_score(y_true, y_score, k=10):
    """Compute Discounted Cumulative Gain (DCG)."""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score_custom(y_true, y_score, k=10):
    """Compute Normalized Discounted Cumulative Gain (nDCG)."""
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0

def mrr_score(y_true, y_score):
    """Compute Mean Reciprocal Rank (MRR)."""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0

def reciprocal_rank(y_true, y_score):
    """Compute Reciprocal Rank (RR) for top-5 results."""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    for i in range(min(5, len(y_true))):  # Only check top 5
        if y_true[i] == 1:
            return 1 / (i + 1)
    return 0

def precision(y_true, y_pred):
    """Calculate precision."""
    tp = sum(yt == 1 and yp == 1 for yt, yp in zip(y_true, y_pred))
    fp = sum(yt == 0 and yp == 1 for yt, yp in zip(y_true, y_pred))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    """Calculate recall."""
    tp = sum(yt == 1 and yp == 1 for yt, yp in zip(y_true, y_pred))
    fn = sum(yt == 1 and yp == 0 for yt, yp in zip(y_true, y_pred))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_at_k(y_true, y_score):
    """Compute F1-score@K."""
    prec = precision(y_true, y_score)
    rec = recall(y_true, y_score)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

# --- Compute Metrics Function ---
def compute_metrics(df, cutoff = 0.5):
    """Compute AUC, MRR, RR, nDCG, Precision, Recall, and F1-score metrics."""
    aucs, mrrs, rrs, ndcg5s, ndcg10s = [], [], [], [], []
    precisions, recalls, f1s = [], [], []
    y_true_all, y_score_all = [], []

    for user_id, group in df.groupby('user_id'):
        y_true = group['label'].values
        y_score = group['similarity'].values

        if len(np.unique(y_true)) < 2:
            continue  # AUC needs both classes

        auc = roc_auc_score(y_true, y_score)
        mrr = mrr_score(y_true, y_score)
        rr = reciprocal_rank(y_true, y_score)
        ndcg5 = ndcg_score_custom(y_true, y_score, k=5)
        ndcg10 = ndcg_score_custom(y_true, y_score, k=10)

        #bin the scores for the binary metrics
        y_score = np.where(np.array(y_score) < cutoff, 0, 1)

        pre = precision(y_true, y_score)
        rec = recall(y_true, y_score)
        f1 = f1_at_k(y_true, y_score)

        aucs.append(auc)
        mrrs.append(mrr)
        rrs.append(rr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)
        precisions.append(pre)
        recalls.append(rec)
        f1s.append(f1)

        y_true_all.extend(y_true)
        y_score_all.extend(y_score)

    class_report = classification_report(y_true_all, y_score_all, digits=4)

    return {
        'AUC': np.mean(aucs),
        'MRR': np.mean(mrrs),
        'RR': np.mean(rrs),
        'nDCG@5': np.mean(ndcg5s),
        'nDCG@10': np.mean(ndcg10s),
        'Precision@10': np.mean(precisions),
        'Recall@10': np.mean(recalls),
        'F1-score@10': np.mean(f1s),
        'Classification Report': class_report
    }

# --- Evaluation Function ---
def evaluate_model(model, df_validation, cutoff = 0.5):
    """Evaluate the model using HR@K, AUC, MRR, and nDCG metrics."""
    results = []

    for _, row in df_validation.iterrows():
        if pd.isna(row["impressions"]):
            continue

        candidate_articles = [news_id for news_id, _ in row["impressions"]]
        ground_truth = {news_id for news_id, click in row["impressions"] if click == 1}

        if not candidate_articles or not ground_truth:
            continue  # Skip cases with no candidates or no clicks

        predicted_scores = model.predict(candidate_articles)
        ranked_articles = predicted_scores.sort_values(ascending=False)

        # Create DataFrame for computing metrics
        user_results = pd.DataFrame({
            'user_id': row['user_id'],
            'news_id': ranked_articles.index,
            'similarity': ranked_articles.values,
            'label': [1 if news_id in ground_truth else 0 for news_id in ranked_articles.index]
        })
        results.append(user_results)

    # Merge results
    df_results = pd.concat(results, ignore_index=True)

    # Compute metrics
    metrics = compute_metrics(df_results, cutoff)

    print(f"📊 Evaluation Results:")
    for metric, value in metrics.items():
        if metric == 'Classification Report':
            print("\nClassification Report:")
            print(value)
        else:
            print(f"{metric}: {value:.4f}")

    # Plot similarity distribution
    plt.figure(figsize=(6, 4))
    df_results['similarity'].hist(bins=50)
    plt.title('Similarity Score Distribution')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.show()

    # Compute and plot ROC curve
    df_results['similarity'] = np.where(df_results['similarity'] < cutoff, 0, 1)
    fpr, tpr, _ = roc_curve(df_results['label'].astype(int), df_results['similarity'])

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")  # Add y = x line
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    tracker = EmissionsTracker()

    # Example usage:
    # Assuming you have the MIND dataset files in a directory called 'data'
    # and want to load the small version of the dataset
    mind_data_loader = MINDDataLoader(
        dataset_path="data"
    )
    mind_data_loader.unzip_files(
        file_name="MINDsmall_train.zip", split="train")
    mind_data_loader.unzip_files(
        file_name="MINDsmall_dev.zip", split="validation")

    # Load user-item interactions
    df_behaviors_train, df_news_train = mind_data_loader.load_interactions(
        split="train")
    df_behaviors_validation, df_news_validation = mind_data_loader.load_interactions(
        split="validation")

    # Preprocess the interactions
    # preprocessor = CollaborativeFilteringPreprocessor()
    preprocessor = BaselinePreprocessor()
    df_behaviors_train = preprocessor.preprocess_interactions(
        df_behaviors_train)
    df_behaviors_validation = preprocessor.preprocess_interactions(
        df_behaviors_validation)

    tracker.start()
    #model = RecencyModel(df_behaviors_train)
    model = PopularityModel(df_behaviors_train)
    model.fit()

    df_behaviors_validation["impressions"] = df_behaviors_validation["impressions"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    #df_behaviors_validation = df_behaviors_validation[:1000]

    # Evaluate the model
    evaluate_model(model, df_behaviors_validation, model.quantile())

    tracker.stop()

