import numpy as np
from scipy.linalg import det, inv

def bhattacharyya_bound(mu1, cov1, mu2, cov2):
    d = len(mu1)

    # Bhattacharyya distance
    bc_distance = 1/8 * np.dot(np.dot((mu1 - mu2).T, inv((cov1 + cov2)/2)), (mu1 - mu2)) + 1/2 * np.log(det((cov1 + cov2)/2) / np.sqrt(det(cov1) * det(cov2)))

    # Bhattacharyya bound
    pe_min = 1/2 * (1 - np.sqrt(1 - np.exp(-bc_distance)))

    return pe_min

def bhattacharyya_bound_1d(mu1, var1, mu2, var2):
    # Bhattacharyya distance
    bc_distance = 1/8 * ((mu1 - mu2)**2 / (var1 + var2)) + 1/2 * np.log((var1 + var2) / (2 * np.sqrt(var1 * var2)))

    # Bhattacharyya bound
    pe_min = 1/2 * (1 - np.sqrt(1 - np.exp(-bc_distance)))

    return pe_min