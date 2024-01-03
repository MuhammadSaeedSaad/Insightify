from discriminants import discriminant_function

def classifier_three_categories(x, mu1, cov1, mu2, cov2, mu3, cov3, prior_probabilityw1, prior_probabilityw2, prior_probabilityw3):
  # g1 for w1 & g2 for w2 & g3 for w3
  g1 = discriminant_function(x, mu1, cov1, prior_probabilityw1)
  g2 = discriminant_function(x, mu2, cov2, prior_probabilityw2)
  g3 = discriminant_function(x, mu3, cov3, prior_probabilityw3)

  discriminants = [g1, g2, g3]
  winner = max(discriminants)
  if winner == g1:
    return "w1"
  elif winner == g2:
    return "w2"
  else:
    return "w3"