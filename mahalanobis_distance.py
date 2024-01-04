import numpy as np

def mahalanobis_distance(x, mean, covariance):
    """
    Calculate the Mahalanobis distance between a point x and the mean with covariance matrix.

    Parameters:
    - x: Input vector (1-D array or list).
    - mean: Mean vector (1-D array or list).
    - covariance: Covariance matrix (2-D array or list).

    Returns:
    - mahalanobis_distance: Mahalanobis distance between x and the mean.
    """
    # Ensure input vectors are numpy arrays
    x = np.array(x)
    mean = np.array(mean)

    # Calculate Mahalanobis distance
    cov_inv = np.linalg.inv(covariance)
    mahalanobis_distance = np.sqrt(np.dot(np.dot((x - mean).T, cov_inv), (x - mean)))

    return mahalanobis_distance

def mahalanobis_distances(x, mus, covs):
    distances = []
    for i in range(len(mus)):
        distances.append(mahalanobis_distance(x, mus[i], covs[i]))
    return distances

# Example usage:
# mean_vector = np.array([1, 2])
# covariance_matrix = np.array([[3, 0.5], [0.5, 1]])
# input_vector = np.array([0.5, 1.5])

# mahalanobis_result = mahalanobis_distance(input_vector, mean_vector, covariance_matrix)
# print("Mahalanobis Distance:", mahalanobis_result)
