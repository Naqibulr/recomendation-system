import pandas as pd
import numpy as np
from pathlib import Path
import zipfile


class MINDDataLoader:
    def __init__(self, dataset_path="data"):
        self.dataset_path = Path(dataset_path)
        self.dataset_path_train = self.dataset_path / "train"
        self.dataset_path_validation = self.dataset_path / "validation"

    def unzip_files(
        self, file_name="MINDsmall_train.zip", split: str = "train", clean_up=False
    ):
        # Unzip files if they are zipped
        zip_path = self.dataset_path / file_name
        destination_path = self.dataset_path / split

        # if folder is already unzipped and the folder is not empty, return
        if destination_path.exists() and len(list(destination_path.iterdir())) > 0:
            print(
                f"The {split} folder already exists and is not empty. Skipping...",
                "\n",
                "If you want to re-unzip the files, delete the folder and try again.",
            )

            return
        print("Unzipping files...")
        print(f"Zip path: {zip_path}")
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(destination_path)
            print("Files unzipped")
            if clean_up:
                zip_path.unlink()
                print("Zip file deleted")
        else:
            print("Cannot find the specified zip file")

    def load_interactions(self, split="train"):
        # Load user-item interactions (e.g., clicks, reads)
        if split == "train":
            df_behaviors = pd.read_csv(
                self.dataset_path_train / "behaviors.tsv", sep="\t", header=None
            )
            df_news = pd.read_csv(
                self.dataset_path_train / "news.tsv", sep="\t", header=None
            )

        elif split == "validation":
            df_behaviors = pd.read_csv(
                self.dataset_path_validation / "behaviors.tsv", sep="\t", header=None
            )
            df_news = pd.read_csv(
                self.dataset_path_validation / "news.tsv", sep="\t", header=None
            )
        else:
            raise ValueError("Split must be either 'train' or 'validation'")

        df_behaviors.columns = [
            "impression_id",
            "user_id",
            "time",
            "history",
            "impressions",
        ]
        df_news.columns = [
            "news_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ]
        return df_behaviors, df_news

    def load_embeddings(self, split="train"):
        if split == "train":
            entity_file = self.dataset_path_train / "entity_embedding.vec"
            relation_file = self.dataset_path_train / "relation_embedding.vec"
        elif split == "validation":
            entity_file = self.dataset_path_validation / "entity_embedding.vec"
            relation_file = self.dataset_path_validation / "relation_embedding.vec"
        else:
            raise ValueError("Split must be either 'train' or 'validation'")

        df_embeddings_entity = np.fromfile(
            str(entity_file), dtype=np.float32
        )  # Replace float32 if needed
        df_embeddings_relation = np.fromfile(
            str(relation_file), dtype=np.float32
        )  # Replace float32 if needed

        return df_embeddings_entity, df_embeddings_relation
