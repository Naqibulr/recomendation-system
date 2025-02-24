import numpy as np


def evaluate_metrics(similarity_matrix):
    """
    Evaluates the cosine similarity matrix by computing example metrics.
    This example computes average similarity per row and overall average similarity.

    Replace or extend these computations with your own evaluation metrics.
    """
    # Average similarity for each sample (assumed rows are samples)
    avg_similarity_per_sample = np.mean(similarity_matrix, axis=1)
    overall_avg_similarity = np.mean(similarity_matrix)

    print("Average similarity per sample:")
    print(avg_similarity_per_sample)

    print("\nOverall average similarity:")
    print(overall_avg_similarity)

    return avg_similarity_per_sample, overall_avg_similarity


def main():
    # Load the cosine similarity matrix from a .npy file
    try:
        similarity_matrix = np.load("cosine_similarity.npy", allow_pickle=True)

    except IOError as ex:
        print("Error loading 'cosine_similarity.npy':", ex)
        return
    
    # Evaluate the similarity matrix
    evaluate_metrics(similarity_matrix)


if __name__ == "__main__":
    main()
