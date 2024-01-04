import numpy as np
from scipy.linalg import inv, det

def discriminant_function(x, mean, covariance, prior_probability):
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

def discriminant_function_1d(x, mean, variance, prior_probability):
    # Ensure input values are converted to numpy arrays
    x = np.array(x)

    # Calculate discriminant function components
    term1 = -0.5 * np.log(2 * np.pi * variance)
    term2 = -0.5 * ((x - mean)**2 / variance)
    term3 = np.log(prior_probability)

    # Calculate discriminant function
    discriminant_value = term1 + term2 + term3

    return discriminant_value
