# Logging, configs, and other utilities


import os

import numpy as np


def save_cosine_similarity_matrix(sim_matrix, filename="cosine_similarity.npy"):
    np.save(filename, sim_matrix)
