import random
import numpy as np

from src.data_loader import MINDDataLoader
from src.evaluation.metrics import EvaluationMetrics
from src.collaborative_filtering.model import CollaborativeFilteringModel
from src.utils import load_interaction_matrix


def main():
    # Load the cosine similarity matrix from a .npy file
    try:
        similarity_matrix = np.load("cosine_similarity.npy", allow_pickle=True)
        interaction_matrix = load_interaction_matrix("interaction_matrix.pkl")
    except IOError as ex:
        print("Error loading files:", ex)
        return

    df_behaviors_train, df_news_train, df_behaviors_validation, df_news_validation = load_data()

    # Try to do some operations with the similarity matrix, like averaging
    # Transform from pickled matrix to a regular numpy array
    similarity_matrix = similarity_matrix.item()
    print("Cosine similarity matrix shape:", similarity_matrix.shape)
    print("Interaction matrix shape:", interaction_matrix.shape)

    model = CollaborativeFilteringModel(df=None)
    model.similarity_matrix = similarity_matrix
    model.interaction_matrix = interaction_matrix

    # pick a random user
    user_id = model.interaction_matrix.index[random.randint(1, 100)]
    print(model.predict(user_id, top_n=10))

    # Evaluate the model
    evaluation = EvaluationMetrics(actual=None, predicted=None)

    # Get list of users present in the validation set and the interaction matrix
    common_users = [user for user in df_behaviors_validation["user_id"].unique(
    ) if user in interaction_matrix.index]

    print("Number of common users:", len(common_users))

    evaluation.actual = [
        [impression.split("-")[0] for impression in df_behaviors_validation[df_behaviors_validation["user_id"] == user]["impressions"].explode()]
        for user in common_users
    ]

    print("Actual values, preview:", evaluation.actual[:5])

    evaluation.predicted = [
        model.predict(user, top_n=10)
        for user in common_users
    ]

    print("Predicted values, preview:", evaluation.predicted[:5])

    print(evaluation.hit_rate_at_k(k=1000))
    print(evaluation.ndcg_at_k(k=1000))
    print(evaluation.precision_at_k(k=1000))
    print(evaluation.recall_at_k(k=1000))


def load_data():
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

    return df_behaviors_train, df_news_train, df_behaviors_validation, df_news_validation


if __name__ == "__main__":
    main()
