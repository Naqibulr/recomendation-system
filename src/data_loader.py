import pandas as pd
import numpy as np
from pathlib import Path
import zipfile

class MINDDataLoader:
    def __init__(self, dataset_path="../data"):
        self.dataset_path = Path(dataset_path)
        self.unzip_files()

    def unzip_files(self, file_name="MINDlarge_train.zip"):
        # Unzip files if they are zipped
        zip_path = self.dataset_path.parent / file_name
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_path.parent)

    def load_interactions(self, split="train"):
        # Load user-item interactions (e.g., clicks, reads)
        if split == "train":
            df = pd.read_csv(self.dataset_path / "train_impressions.csv")
        elif split == "validation":
            df = pd.read_csv(self.dataset_path / "validation_impressions.csv")
        else:
            raise ValueError("Split must be either 'train' or 'validation'")
        return df

    def load_item_features(self):
        # Load article metadata/text embeddings
        df = pd.read_csv(self.dataset_path / "news_articles.csv")
        embeddings = np.load(self.dataset_path / "text_embeddings.npy")
        df["embedding"] = list(embeddings)  # Attach embeddings to items
        return df
