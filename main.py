from src.data_loader import MINDDataLoader

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


    # Check that the data is loaded correctly
    print(df_behaviors_train.head())
    print(df_news_train.head())

    print(df_behaviors_validation.head())
    print(df_news_validation.head())
