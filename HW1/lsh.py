# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla
#Modified: Alex Porter
import numpy as np
import random
import time
import pdb
import unittest
import matplotlib.pyplot as plt
from PIL import Image

# Finds the L1 distance between two vectors
# u and v are 1-dimensional np.array objects
# TODO: Implement this
def l1(u, v):
    res = 0
    for i in range(len(u)):
        res += max(u[i] - v[i], v[i] - u[i])
    return res
    # raise NotImplementedError

# Loads the data into a np array, where each row corresponds to
# an image patch -- this step is sort of slow.
# Each row in the data is an image, and there are 400 columns.
def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

# Creates a hash function from a list of dimensions and thresholds.
def create_function(dimensions, thresholds):
    def f(v):
        boolarray = [v[dimensions[i]] >= thresholds[i] for i in range(len(dimensions))]
        return "".join(map(str, map(int, boolarray)))
    return f

# Creates the LSH functions (functions that compute L K-bit hash keys).
# Each function selects k dimensions (i.e. column indices of the image matrix)
# at random, and then chooses a random threshold for each dimension, between 0 and
# 255.  For any image, if its value on a given dimension is greater than or equal to
# the randomly chosen threshold, we set that bit to 1.  Each hash function returns
# a length-k bit string of the form "0101010001101001...", and the L hash functions 
# will produce L such bit strings for each image.
def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    functions = []
    for i in range(L):
        dimensions = np.random.randint(low = 0, 
                                   high = num_dimensions,
                                   size = k)
        thresholds = np.random.randint(low = min_threshold, 
                                   high = max_threshold + 1, 
                                   size = k)

        functions.append(create_function(dimensions, thresholds))
    return functions

# Hashes an individual vector (i.e. image).  This produces an array with L
# entries, where each entry is a string of k bits.
def hash_vector(functions, v):
    return np.array([f(v) for f in functions])

# Hashes the data in A, where each row is a datapoint, using the L
# functions in "functions."
def hash_data(functions, A):
    return np.array(list(map(lambda v: hash_vector(functions, v), A)))

# Retrieve all of the points that hash to one of the same buckets 
# as the query point.  Do not do any random sampling (unlike what the first
# part of this problem prescribes).
# Don't retrieve a point if it is the same point as the query point.
def get_candidates(hashed_A, hashed_point, query_index):
    return filter(lambda i: i != query_index and \
        any(hashed_point == hashed_A[i]), range(len(hashed_A)))

# Sets up the LSH.  You should try to call this function as few times as 
# possible, since it is expensive.
# A: The dataset in which each row is an image patch.
# Return the LSH functions and hashed data structure.
def lsh_setup(A, k = 24, L = 10):
    functions = create_functions(k = k, L = L)
    hashed_A = hash_data(functions, A)
    return (functions, hashed_A)

# Run the entire LSH algorithm
def lsh_search(A, hashed_A, functions, query_index, num_neighbors = 10):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)
    
    distances = map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[0] for t in best_neighbors]

# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")

# Finds the nearest neighbors to a given vector, using linear search.
def linear_search(A, query_index, num_neighbors):
    #TODO
    distances = []
    for i, point in enumerate(A):
        if(i == query_index):
            continue
        distances.append((i, l1(A[query_index], point)))
    best_neighbors = sorted(distances, key=lambda x: x[1])[:num_neighbors]
    return [t[0] for t in best_neighbors]

# TODO: Write a function that computes the error measure
def error_measure(A):
    points = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    Ls = [10, 12, 14, 16, 18, 20]
    Ks = [16, 18, 20, 22, 24]
    l_errors = []
    for l in Ls:
        sum_l = 0
        functions, hashed_A = lsh_setup(A, 24, l)
        for z in points:
            LSH_neighbors = lsh_search(A, hashed_A, functions, z, 3)
            linear_neighbors = linear_search(A, z, 3)
            num, den = 0, 0
            for i in range(3):
                num += l1(A[LSH_neighbors[i]], A[z])
                den += l1(A[linear_neighbors[i]], A[z])
        sum_l += (num / den)
        l_errors.append(sum_l / 10)

    k_errors = []
    for k in Ks:
        sum_k = 0
        functions, hashed_A = lsh_setup(A, k, 10)
        for z in points:
            LSH_neighbors = lsh_search(A, hashed_A, functions, z, 3)
            linear_neighbors = linear_search(A, z, 3)
            num, den = 0, 0
            for i in range(3):
                num += l1(A[LSH_neighbors[i]], A[z])
                den += l1(A[linear_neighbors[i]], A[z])
        sum_k += (num / den)
        k_errors.append(sum_k / 10)

    return l_errors, k_errors

# TODO: Solve Problem 4
def problem4():
    A = load_data("./data/patches.csv")
    functions, hashed_A = lsh_setup(A)
    points = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    lsh_times = []
    linear_times = []
    for point in points:
        start_lsh = time.time()
        print(f"3 nearest neighbors for {point} (LSH):\t{lsh_search(A, hashed_A, functions, point, 3)}")
        end_lsh = time.time()
        lsh_times.append(end_lsh - start_lsh)

        start_linear = time.time()
        print(f"3 nearest neighbors for {point} (LINEAR):\t{linear_search(A, point, 3)}\n")
        end_linear = time.time()
        linear_times.append(end_linear - start_linear)

    # Average Times
    print("Average time (LSH) = ", sum(lsh_times) / 10)
    print("Average time (Linear) = ", sum(linear_times) / 10)

    # Error plots
    Ls = [10, 12, 14, 16, 18, 20]
    Ks = [16, 18, 20, 22, 24]
    l_errors, k_errors = error_measure(A)
    plt.plot(Ls, l_errors)
    plt.savefig("L.png")
    plt.plot(Ks, k_errors)
    plt.savefig("K.png")

    # Plots
    plot(A, [100], 'original')
    plot(A, lsh_search(A, hashed_A, functions, 100), "100-lsh")
    plot(A, linear_search(A, 100, 10), "100-linear")

    #raise NotImplementedError

#### TESTS #####

class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0,:]), 6)
        self.assertEqual(f2(A[0,:]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))

    ### TODO: Write your tests here (they won't be graded, 
    ### but you may find them helpful)


if __name__ == '__main__':
#    unittest.main() ### TODO: Uncomment this to run tests
    #tester = TestLSH()
    #tester.test_l1()
    problem4()
    
