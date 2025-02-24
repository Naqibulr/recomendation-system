from src.collaborative_filtering.model import CollaborativeFilteringModel
from src.data_loader import MINDDataLoader
from src.collaborative_filtering.preprocessing import CollaborativeFilteringPreprocessor
from src.evaluation.metrics import EvaluationMetrics
from src.utils import save_cosine_similarity_matrix


def main():
    print("Running recommendation systems...")
    # Add code to run different filtering recommender systems here
    pass


if __name__ == "__main__":
    main()
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
    preprocessor = CollaborativeFilteringPreprocessor()
    df_behaviors_train = preprocessor.preprocess_interactions(
        df_behaviors_train)

    small_df = df_behaviors_train.sample(10000)  # Reduce dataset
    model = CollaborativeFilteringModel(small_df)
    model.fit()

    sample_user_id = model.interaction_matrix.index[4]  # Just an example user

    # Get recommendations
    print(model.similarity_matrix)
    save_cosine_similarity_matrix(model.similarity_matrix)
    recommendations = model.predict(sample_user_id)

    evaluation = EvaluationMetrics(actual=None, predicted=None
                                   )
