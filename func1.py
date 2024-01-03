import numpy as np

def generate_multivariate_normal_samples(mean, covariance_matrix, num_samples):
    """
    Generate random samples from a multivariate normal distribution.

    Parameters:
    - mean: Mean vector (1-D array) of length d.
    - covariance_matrix: Covariance matrix (2-D array) of shape (d, d).
    - num_samples: Number of samples to generate.

    Returns:
    - samples: 2-D array of shape (num_samples, d) containing generated samples.
    """
    samples = np.random.multivariate_normal(mean, covariance_matrix, num_samples)
    return samples

# Example usage:
mean_vector = np.array([0, 1])
covariance_matrix = np.array([[1, 0.5], [0.5, 2]])
num_samples = 100

generated_samples = generate_multivariate_normal_samples(mean_vector, covariance_matrix, num_samples)
print(generated_samples)
