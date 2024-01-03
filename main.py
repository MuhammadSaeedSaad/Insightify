from func2 import discriminant_function, discriminant_function_1d
from dataset import dataset
import numpy as np
from bhattacharyya import bhattacharyya_bound, bhattacharyya_bound_1d
from dichotomizer import dichotomizer, dichotomizer_1d

# Q2-b 1 dimension
pw1 = 0.5
pw2 = 0.5
first_column_w1 = [row[0] for row in dataset[0]]
first_column_w2 = [row[0] for row in dataset[1]]
mu1 = np.mean(first_column_w1)
mu2 = np.mean(first_column_w2)
var1 = np.cov(first_column_w1, bias=True)
var2 = np.cov(first_column_w2, bias=True)

error_count = 0
for i in range(len(first_column_w1)):
  result = dichotomizer_1d(dataset[0][i][0], mu1, var1, mu2, var2, pw1, pw2)
  if result != "w1":
    error_count += 1

for i in range(len(first_column_w2)):
  result = dichotomizer_1d(dataset[1][i][0], mu1, var1, mu2, var2, pw1, pw2)
  if result != "w2":
    error_count += 1
error_percent = error_count / (len(first_column_w2) * 2)
print(error_percent)  # 0.35

# Q2-c
bhattacharyya_bound_1d(mu1, var1, mu2, var2)

# Q2-d
first_2_cols_w1 = [[row[0], row[1]] for row in dataset[0]]