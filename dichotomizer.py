from func2 import discriminant_function, discriminant_function_1d
from dataset import dataset
import numpy as np
from bhattacharyya import bhattacharyya_bound, bhattacharyya_bound_1d

def dichotomizer(x, mean1, covariance1, mean2, covariance2, prior_probabilitw1, prior_probabilitw2):
  # g1 for w1 & g2 for w2
  g1 = discriminant_function(x, mean1, covariance1, prior_probabilitw1)
  g2 = discriminant_function(x, mean2, covariance2, prior_probabilitw2)

  g = g1 - g2

  if g > 0:
    return "w1"
  else:
    return "w2"
  
def dichotomizer_1d(x, mean1, covariance1, mean2, covariance2, prior_probabilitw1, prior_probabilitw2):
  # g1 for w1 & g2 for w2
  g1 = discriminant_function_1d(x, mean1, covariance1, prior_probabilitw1)
  g2 = discriminant_function_1d(x, mean2, covariance2, prior_probabilitw2)

  g = g1 - g2

  if g > 0:
    return "w1"
  else:
    return "w2"



# mean_vector1 = np.mean(dataset[0], axis=0)
# mean_vector2 = np.mean(dataset[1], axis=0)
# mean_vector3 = np.mean(dataset[2], axis=0)
# print(mean_vector1)

# cov_matrix1 = np.cov(dataset[0], rowvar=False, bias=True)
# cov_matrix2 = np.cov(dataset[1], rowvar=False, bias=True)
# cov_matrix3 = np.cov(dataset[2], rowvar=False, bias=True)
# print(cov_matrix1)