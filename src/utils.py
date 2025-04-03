# Logging, configs, and other utilities
import numpy as np
import pandas as pd


def save_cosine_similarity_matrix(sim_matrix, filename="cosine_similarity.npy"):
    np.save(filename, sim_matrix)


def save_interaction_matrix(interaction_matrix, file_path):
    interaction_matrix.to_pickle(file_path)


def load_interaction_matrix(file_path):
    return pd.read_pickle(file_path)
