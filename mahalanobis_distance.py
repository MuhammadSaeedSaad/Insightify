import numpy as np

def mahalanobis_distance(x, mean, covariance):
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
