from discriminants import discriminant_function

def classifier_three_categories(x, mus, covs, priors):
  # g1 for w1 & g2 for w2 & g3 for w3
  discriminants = []
  num_cats = len(mus)
  for i in range(num_cats):
    discriminants.append(discriminant_function(x, mus[i], covs[i], priors[i]))
  winner = discriminants.index(max(discriminants))
  # increase 1 to the winner as we count from 0 here and count from 1 in the classes names
  return "w" + str(winner + 1)