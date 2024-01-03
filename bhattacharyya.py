import numpy as np
from scipy.linalg import det, inv

def bhattacharyya_bound(mu1, cov1, mu2, cov2):
    """
    Calculate Bhattacharyya bound for two Gaussian distributions.

    Parameters:
    - mu1: Mean vector of the first distribution (1-D array).
    - cov1: Covariance matrix of the first distribution (2-D array).
    - mu2: Mean vector of the second distribution (1-D array).
    - cov2: Covariance matrix of the second distribution (2-D array).

    Returns:
    - bhattacharyya_bound: Bhattacharyya bound.
    """
    d = len(mu1)

    # Bhattacharyya distance
    bc_distance = 1/8 * np.dot(np.dot((mu1 - mu2).T, inv((cov1 + cov2)/2)), (mu1 - mu2)) + 1/2 * np.log(det((cov1 + cov2)/2) / np.sqrt(det(cov1) * det(cov2)))

    # Bhattacharyya bound
    pe_min = 1/2 * (1 - np.sqrt(1 - np.exp(-bc_distance)))

    return pe_min

import numpy as np
from scipy.linalg import det, inv

def bhattacharyya_bound_1d(mu1, var1, mu2, var2):
    """
    Calculate Bhattacharyya bound for two Gaussian distributions in one dimension.

    Parameters:
    - mu1: Mean of the first distribution.
    - var1: Variance of the first distribution.
    - mu2: Mean of the second distribution.
    - var2: Variance of the second distribution.

    Returns:
    - bhattacharyya_bound: Bhattacharyya bound.
    """
    # Bhattacharyya distance
    bc_distance = 1/8 * ((mu1 - mu2)**2 / (var1 + var2)) + 1/2 * np.log((var1 + var2) / (2 * np.sqrt(var1 * var2)))

    # Bhattacharyya bound
    pe_min = 1/2 * (1 - np.sqrt(1 - np.exp(-bc_distance)))

    return pe_min

# Example usage:
mean1_1d, var1_1d = 1, 2
mean2_1d, var2_1d = 3, 4

bound_1d = bhattacharyya_bound_1d(mean1_1d, var1_1d, mean2_1d, var2_1d)
print(f"Bhattacharyya Bound (1D): {bound_1d}")


# # Example usage:
# mean1 = np.array([1, 2])
# covariance1 = np.array([[2, 1], [1, 3]])
# mean2 = np.array([3, 4])
# covariance2 = np.array([[4, 2], [2, 5]])

# bound = bhattacharyya_bound(mean1, covariance1, mean2, covariance2)
# print(f"Bhattacharyya Bound: {bound}")