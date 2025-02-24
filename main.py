from src.collaborative_filtering.model import CollaborativeFilteringModel
from src.data_loader import MINDDataLoader
from src.collaborative_filtering.preprocessing import CollaborativeFilteringPreprocessor
from src.evaluation.metrics import evaluate_model

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
    mind_data_loader.unzip_files(file_name="MINDsmall_train.zip", split="train")
    mind_data_loader.unzip_files(file_name="MINDsmall_dev.zip", split="validation")

    # Load user-item interactions
    df_behaviors_train, df_news_train = mind_data_loader.load_interactions(split="train")
    df_behaviors_validation, df_news_validation = mind_data_loader.load_interactions(split="validation")

    # Preprocess the interactions
    df_behaviors_train = CollaborativeFilteringPreprocessor.preprocess_interactions(df_behaviors_train)
    print(df_behaviors_train.head())

    # debug print impressions to see whether format is correct.
    print(df_behaviors_train['impressions'])

        # Create and fit the collaborative filtering model
    #model = CollaborativeFilteringModel(df_behaviors_train)
    #model.fit()

    """

    small_df = df_behaviors_train.sample(10000)  # Reduce dataset
    model = CollaborativeFilteringModel(small_df)
    model.fit()

    #print(model.interaction_matrix.head())
    #print(model.similarity_matrix.shape)


    # Example prediction (assuming you have a predict method)
    #user_id = df_behaviors_train['user_id'].iloc[0]  # Get the first user ID
    #predictions = model.predict(user_id)
    #print(f"Predictions for user {user_id}: {predictions}")

    # Pick a user ID from the dataset
    sample_user_id = model.interaction_matrix.index[4]  # Just an example user
    print(f"Getting recommendations for user ID: {sample_user_id}")

    # Get recommendations
    recommendations = model.predict(sample_user_id)
    print(recommendations)

    print("Eval results")
    evaluation_results = evaluate_model(
        recommendations, df_behaviors_validation
    )
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value}")


"""