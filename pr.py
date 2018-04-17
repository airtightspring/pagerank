import numpy as np
import matplotlib as plt

# data input: setting up constants
alpha_constant = 0.85
epsilon_constant = 0.00001

# creating an adjacency matrix
matrix = np.matrix([[1.0, 0, 2, 0, 4, 3], [3, 0, 1, 1, 0, 0], [2, 0, 4, 0, 1, 0], [0, 0, 1, 0, 0, 1], [8, 0, 3, 0, 5, 2]
                    , [0, 0, 0, 0, 0, 0]])

# modifying the adjacency matrix
# set all diagonals to 0
np.fill_diagonal(matrix, 0)
# sum columns
column_sum = np.sum(matrix, 0)
matrix_dims = matrix.shape
# divide numbers by sum of columns
for i in range(0, matrix_dims[0]):
    for j in range(0, matrix_dims[1]):
        if matrix[i, j] != 0 and column_sum[0, j] != 0:
            matrix[i, j] = matrix[i, j] / column_sum[0, j]

# identify dangling nodes
new_column_sum = np.sum(matrix, 0)
new_column_sum_size = new_column_sum.shape

d = []
for a in range(0, new_column_sum_size[1]):
    if new_column_sum[0, a] != 0:
        d.append(0)
    else:
        d.append(1)

# calculating the influence vector
# calculating the article vector
a = [3, 2, 5, 1, 2, 1]
a_total = 14
a_revised = []
for b in range(0, len(a)):
    a_revised.append(a[b] / a_total)
# calculating initial start vector
i = []
for c in range(0, 6):
    i.append(1 / 6)

# calculating the influence vector
pi = np.array(i).transpose()
sum_pi = pi

next_pi = (0.85 * np.matmul(matrix, pi)) + (0.85 * (np.dot(d, pi)) + 0.15) * np.array(a_revised)

pi = []
for z in range(0, len(np.array(next_pi)[0])):
    pi.append(next_pi[0, z])
pi = np.transpose(pi)
next_pi = (0.85 * np.matmul(matrix, pi)) + (0.85 * (np.dot(d, pi)) + 0.15) * np.array(a_revised)

while np.linalg.norm((next_pi - pi), ord=1) > epsilon_constant:
    pi = []
    for z in range(0, len(np.array(next_pi)[0])):
        pi.append(next_pi[0, z])
    pi = np.transpose(pi)
    next_pi = (0.85 * np.matmul(matrix, pi)) + (0.85 * (np.dot(d, pi)) + 0.15) * np.array(a_revised)
    sum_pi = sum_pi + pi

# calculating EigenFactor

pi = []
for z in range(0, len(np.array(next_pi)[0])):
    pi.append(next_pi[0, z])
pi = np.transpose(pi)

eigen_vals = np.dot(matrix, pi) / np.sum(np.dot(matrix, pi)) * 100

