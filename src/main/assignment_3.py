import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

# Problem 1
# Euler Method with the following details
# a. Function: t - y^2
# b. Range: 0 < t < 2
# c. Iterations: 10
# d. Initial Point: f(0) = 1

def function(t: float, y: float):
  return t - (y**2)

def eulers_method(initialCondition, startOfT, endOfT, iterations):
  h = (endOfT - startOfT) / iterations
  t, w = startOfT, initialCondition

  for i in range(iterations):
    w = w + (h * (function(t, w)))
    t += h

  return w


# Problem 2
# Runge-Kutta with the following details:
# a. Function: t - y^2
# b. Range: 0 < t < 2
# c. Iterations: 10
# d. Initial Point: f(0) = 1
def runge_kutta(initial_condition, startOfT, endOfT, iterations):
  h = (endOfT - startOfT) / iterations
  t, w = startOfT, initial_condition

  for i in range(iterations):
    order1 = h * function(t, w)
    order2 = h * function(t + (h / 2), w + (order1 / 2))
    order3 = h * function(t + (h / 2), w + (order2 / 2))
    order4 = h * function(t + h, w + order3)

    w = w + ((order1 + (2 * order2) + (2 * order3) + order4) / 6)
    t += h

  return w

# Problem 3
# Use gaussian elimination and backward substitution solve the following linear system of equations written in augmented matrix format
#  2  -1   1  |  6
#  1   3   1  |  0
# -1   5   4  | -3 
def gauss_jordan(matrix):
  length = len(matrix)
  list = []

  for i in range(length):
    maxRow = i
    for j in range(i + 1, length):
      if abs(matrix[j][i]) > abs(matrix[maxRow][i]):
        maxRow = j

    matrix[[i, maxRow]] = matrix[[maxRow, i]]

    pivot = matrix[i][i]
    for j in range(i, length + 1):
      matrix[i][j] /= pivot
        
    for j in range(i + 1, length):
      factor = matrix[j, i]
      for k in range(length + 1):
        matrix[j][k] -= (factor * matrix[i][k])

  for i in range(length - 1, -1, -1):
    for j in range(i - 1, -1, -1):
      factor = matrix[j, i]
      for k in range(length + 1):
        matrix[j][k] -= (factor * matrix[i][k])

  for i in range(length):
    list.append(matrix[i][length])
    
  return list

# Problem 4
# Implement LU Factorization for the following matrix and do the following
#  1  1  0  3
#  2  1 -1  1
#  3 -1 -1  2
# -1  2  3 -1
# a. Print out the matrix determinant
# b. Print out the L matrix
# c. Print out the U matrix
def matrixDeterminant(matrix):
 return np.linalg.det(matrix)

def LUdecomposition(matrix):
  length = len(matrix)
  L = np.zeros((length, length))
  U = np.zeros((length, length))

  # Make LU Matrices
  for i in range(0, length):
    L[i][i] = 1

    # Make Upper Triangle Matrix
    for j in range(i, length):
      U[i][j] = matrix[i][j]
      for k in range(0, i):
        U[i][j] -= L[i][k] * U[k][j]

    # Make Lower Triangle Matrix
    for j in range(i + 1, length):
      L[j][i] = matrix[j][i]
      for k in range(0, i):
        L[j][i] -= L[j][k] * U[k][i]
      L[j][i] /= U[i][i]

  return L, U

# Problem 5
# Determine if the following matrix is diagonally dominate
#  9  0  5  2  1
#  3  9  1  2  1
#  0  1  7  2  3
#  4  2  3 12  2
#  3  2  4  0  8
def diagonally_dominant(matrix):
  sum = 0
  length = len(matrix)
  for i in range(length):
    sum = sum - abs(matrix[i][i])
    for j in range(length):
      sum = sum + abs(matrix[i][j])
        
    if (abs(matrix[i][i]) < sum):
      return False
  return True

# Problem 6
# Determine if the matrix is a positive definite
#  2  2  1
#  2  3  0
#  1  0  2
def positive_definitive(matrix):
  eigenvalues = np.linalg.eigvals(matrix)
  for val in eigenvalues:
    if val <= 0:
      return False
    
  return True

if __name__ == "__main__":
  initial_condition = 1
  start_of_t, end_of_t = 0, 2
  iterations = 10
  print("%.5f" % eulers_method(initial_condition, start_of_t, end_of_t, iterations))
  print()

  print("%.5f" % runge_kutta(initial_condition, start_of_t, end_of_t, iterations))
  print()

  matrix = np.array([[2., -1., 1., 6.], [1., 3., 1., 0.], [-1., 5., 4., -3.]])
  print(np.array(gauss_jordan(matrix)))
  print()

  matrix = np.array([[1., 1., 0., 3.], [2., 1., -1., 1.], [3., -1., -1., 2.], [-1., 2., 3., -1.]])
  print("%.5f" % np.array(matrixDeterminant(matrix)))
  L, U = LUdecomposition(matrix)
  print()
  print(L)
  print()
  print(U)
  print()
  
  matrix = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 7, 2], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]])
  print(diagonally_dominant(matrix))
  print()

  matrix = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
  print(positive_definitive(matrix))