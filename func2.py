import numpy as np
from scipy.linalg import inv, det

def discriminant_function(x, mean, covariance, prior_probability):
    """
    Calculate the discriminant function for a given normal distribution and prior probability.

    Parameters:
    - x: Input vector (1-D array).
    - mean: Mean vector (1-D array).
    - covariance: Covariance matrix (2-D array).
    - prior_probability: Prior probability for the class.

    Returns:
    - discriminant_value: Discriminant function value.
    """
    # Ensure input vectors are numpy arrays
    x = np.array(x)
    mean = np.array(mean)

    # Calculate discriminant function components
    cov_inv = inv(covariance)
    term1 = -0.5 * np.log(det(covariance))
    term2 = -0.5 * np.dot(np.dot((x - mean).T, cov_inv), (x - mean))
    term3 = np.log(prior_probability)

    # Calculate discriminant function
    discriminant_value = term1 + term2 + term3

    return discriminant_value

# Example usage:
mean_vector = np.array([1, 2])
covariance_matrix = np.array([[3, 0.5], [0.5, 1]])
prior_probability = 0.5
input_vector = np.array([0.5, 1.5])

discriminant_result = discriminant_function(input_vector, mean_vector, covariance_matrix, prior_probability)
print("Discriminant Function Value:", discriminant_result)
