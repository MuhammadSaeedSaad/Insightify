from discriminants import discriminant_function, discriminant_function_1d

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