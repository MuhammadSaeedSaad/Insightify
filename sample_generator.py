import numpy as np

def generate_multivariate_normal_samples(mean, covariance_matrix, num_samples):
    samples = np.random.multivariate_normal(mean, covariance_matrix, num_samples)
    return samples

# Example
mean_vector = np.array([0, 1])
covariance_matrix = np.array([[1, 0.5], [0.5, 2]])
num_samples = 100

generated_samples = generate_multivariate_normal_samples(mean_vector, covariance_matrix, num_samples)
print(generated_samples)
