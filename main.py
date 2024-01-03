from dataset import dataset
import numpy as np
from bhattacharyya import bhattacharyya_bound, bhattacharyya_bound_1d
from dichotomizer import dichotomizer, dichotomizer_1d
from classifier_three_categories import classifier_three_categories
from test_points import test_points

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
# print(error_percent)  # 0.35



# Q2-c
# bhattacharyya_bound_1d(mu1, var1, mu2, var2)



# Q2-d
first_2_cols_w1 = [[row[0], row[1]] for row in dataset[0]]
first_2_cols_w2 = [[row[0], row[1]] for row in dataset[1]]
mmu1 = np.mean(first_2_cols_w1, axis=0)
mmu2 = np.mean(first_2_cols_w2, axis=0)
ccov1 = np.cov(first_2_cols_w1, rowvar=False, bias=True)
ccov2 = np.cov(first_2_cols_w2, rowvar=False, bias=True)

error_count2 = 0
for i in range(len(first_column_w1)):
  result = dichotomizer(first_2_cols_w1[i], mmu1, ccov1, mmu2, ccov2, pw1, pw2)
  if result != "w1":
    error_count2 += 1

for i in range(len(first_column_w2)):
  result = dichotomizer(first_2_cols_w2[i], mmu1, ccov1, mmu2, ccov2, pw1, pw2)
  if result != "w2":
    error_count2 += 1
error_percent2 = error_count2 / (len(first_column_w2) * 2)
# print(error_percent2)


# Q2-e
mmmu1 = np.mean(dataset[0], axis=0)
mmmu2 = np.mean(dataset[1], axis=0)
cccov1 = np.cov(dataset[0], rowvar=False, bias=True)
cccov2 = np.cov(dataset[1], rowvar=False, bias=True)

error_count3 = 0
for i in range(len(first_column_w1)):
  result = dichotomizer(dataset[0][i], mmmu1, cccov1, mmmu2, cccov2, pw1, pw2)
  # print(result)
  if result != "w1":
    error_count2 += 1

for i in range(len(first_column_w2)):
  result = dichotomizer(dataset[1][i], mmmu1, cccov1, mmmu2, cccov2, pw1, pw2)
  print(result)
  if result != "w2":
    error_count2 += 1
error_percent3 = error_count3 / (len(first_column_w2) * 2)
print(error_percent3)

# Q2-f
# DISCUSS ALL PREVIOUS RESULTS. 
# Yes, Increasing the dimensions doen't mean necessairly decreasing the error as we saw in the previous step Q2-d we used 2 dimensions insead of 1
# and the error increase from 0.35 when using 1 dimension to 0.4 when using 2.

# Q5-a

# mean_vector1 = np.mean(dataset[0], axis=0)
# mean_vector2 = np.mean(dataset[1], axis=0)
# mean_vector3 = np.mean(dataset[2], axis=0)
# print(mean_vector1)

# cov_matrix1 = np.cov(dataset[0], rowvar=False, bias=True)
# cov_matrix2 = np.cov(dataset[1], rowvar=False, bias=True)
# cov_matrix3 = np.cov(dataset[2], rowvar=False, bias=True)
# print(cov_matrix1)