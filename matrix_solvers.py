# Ethan Watanabe
# NE 155 Homework 2
# Question 4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def gaussian_elimination(A, b):
    """
    Will find the solution vector in a system of linear equations using Gaussian Elimination. 

    :param A (np.ndarray): a nxn matrix
    :param b (np.ndarray): a vector with n components
    """
    # making sure that A is an nxn matrix and b is a 1xn vector
    try:
        np.dot(A,b)
    except:
        raise Exception("Make sure that A is an nxn matrix and that b is a vector with n elements")
    rows, cols = np.shape(A) # nxn so rows == cols
    aug_M = np.hstack([A, b.reshape([rows,1])])
    # row reduction will only work properly if the elements of the matrix and vector are floats
    aug_M = aug_M.astype(float) 
    b = b.astype(float)

    # scaling factor
    s = np.zeros(cols)
    for i in range(rows):
        s[i] = np.max(np.abs(aug_M[i, :]))

    # let's start with row swapping
    # we want row with the highest coefficient relative to its other coefficients to be first
    for k in range(cols):
        # find max value in k column
        p = np.max(np.abs(aug_M[:, k]) / s)
        p_index = np.abs(aug_M[:, k]) / s == p
        p_index = np.arange(rows)[p_index][0]
        
        # if p_index is already at the kth row (p_index == k), then we can move on 
        if p_index != k:
            new_M = aug_M.copy()
            better_row = aug_M[p_index, :]

            new_M[k, :] = better_row # replace the top row with the better row
            # print(f"Row {k} has been replaced with row {p_index}")
            # print(new_M)

            new_M[p_index, :] = aug_M[k,:] # now we can replace the old good row with the bad top row
            # time.sleep(2)
            # print(f"Row {p_index} has been replaced with row {k}")

            # aug_M = new_M.astype(float)
            # time.sleep(2)
            # print(new_M)
            # print("---------------------------------------")
            # time.sleep(2)

            # recalculate s
            s = np.zeros(cols)
            for i in range(rows):
                s[i] = np.max(np.abs(aug_M[i, :]))

        for l in range(k + 1, rows):
            mult_factor = aug_M[l, k] / aug_M[k, k]

            aug_M[l, :] = aug_M[l, :] - aug_M[k, :] * mult_factor
        # print(aug_M)
        # time.sleep(2)
    print(aug_M)
    # back substitution
    # works under the assumption that aug_M is now an upper triangular matrix with some row reduced b slapped onto it
    sol = np.zeros(rows)
    for i in range(1, rows+1):
        sol[-1*i] = aug_M[-1*i, -1] # start b_i
        # print(f"We start with {aug_M[-1*i, -1]}")
        for j in range(cols):
            if j == rows - i:
                pass
            else:
                sol[-1*i] -= aug_M[-1*i, j] * sol[j] # subtract: b_i - a_i{-i j}*x_j
                # print(f"Now we subtract what we currently have with {aug_M[-1*i, j]}*{sol[j]}")
        sol[-1*i] /= aug_M[-1*i, -1*i - 1]
        # print(f"This results in solution {3 - i} to be: {sol[-1*i]}")
    return sol

def jacobi_iteration(A, b, guess, tolerance=1e-8):
    pass

def gauss_seidel(A, b, guess, tolerance=1e-8):
    pass

def sor(A, b, guess, omega, tolerance=1e-8):
    pass

def two_norm(vector):
    output = 0
    for num in vector:
        output += num**2
    output = np.sqrt(output)
    return output

if __name__ == '__main__':
    A = np.array([
        [4, -1,  0,  0,  0],
        [-1, 4, -1,  0,  0],
        [0, -1,  4, -1,  0],
        [0,  0, -1,  4, -1],
        [0,  0,  0, -1,  4]
    ])
    b = np.array([100, 100, 100, 100, 100])
    print(gaussian_elimination(A,b))