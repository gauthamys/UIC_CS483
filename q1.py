import scipy.linalg
import numpy as np

M = np.array([[1, 2], 
              [2, 1], 
              [3, 4], 
              [4, 3]])
U, sigma, Vt = scipy.linalg.svd(M, full_matrices=False)
print(U, sigma, Vt)

assert sigma[0] > sigma[1]
# The first value is greater than the second

mtm = np.matmul(M.transpose(), M)

Evals, Evecs = scipy.linalg.eigh(mtm)

a = list(zip(Evals, Evecs))
b = list(sorted(a, key=lambda x: -x[0]))

Evals_sorted = np.array([x[0] for x in b])
Evecs_sorted = np.array([x[1] for x in b])

print("Sorted Evals:\n", Evals_sorted)
print("Sorted Evecs:\n", Evecs_sorted)

print(Vt.transpose())
print(Evecs_sorted)
# the vectors in Evecs_sorted are scalar multiples of columns of V

print(sigma)
print(Evals_sorted)
# Evals_sorted contains the squares of sigma