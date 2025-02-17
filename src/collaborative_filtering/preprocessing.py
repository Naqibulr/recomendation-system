# Preprocessing functions for collaborative filtering

import pandas as pd

class CollaborativeFilteringPreprocessor:
    def preprocess_interactions(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [
            "impression_id",
            "user_id",
            "time",
            "history",
            "impressions",
        ]

        df["time"] = pd.to_datetime(df["time"])
        df["impressions"] = df["impressions"].apply(
            lambda x: tuple(
                (news_click.split("-")[0], int(news_click.split("-")[1]))
                for news_click in x.split()
            )
        )


        # find duplicate rows
        duplicate_rows = df[df.duplicated()]
        print("Duplicate Rows except first occurrence based on all columns are :")
        print(duplicate_rows)

        # drop duplicate rows
        df = df.drop_duplicates()

        # find rows with missing values
        missing_values = df[df.isnull().any(axis=1)]
        
        print("Rows with missing values:")
        print(missing_values)

        # drop rows with missing values
        df = df.dropna()


        return df


