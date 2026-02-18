# Ethan Watanabe
# NE 155 Homework 2
# Question 4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def gaussian_elimination(A, b):
    """
    Gaussian Elimination solver for a system Ax = b

    :param A (numpy.ndarray): a nxn matrix
    :param b (numpy.ndarray): a vector with n components
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
    """
    Jacobi iteration solver for a system Ax = b.

    Will take a guess and try to converge towards the solution within a tolerance.

    Args:
        A (numpy.ndarray): a nxn matrix 
        b (numpy.ndarray): a vector with n components
        guess (numpy.ndarray): a vector with n components
        tolerance (float): a float of the error tolerance in the iterative method

    Returns:
        k (int): The number of iterations it took to reach a value within tolerance
        next_iter (numpy.ndarray): x^{(k)} the approximate solution for the system
    """
    # making sure that A is an nxn matrix and b is a 1xn vector
    try:
        np.dot(A,b)
    except:
        raise Exception("Make sure that A is an nxn matrix and that b is a vector with n elements")
    A = A.astype(float)
    b = b.astype(float)
    guess = guess.astype(float)
    rows, cols = np.shape(A)

    # lets decompose A into L, D, and U
    L = np.zeros([rows, cols])
    D = np.zeros([rows, cols])
    D_inv = np.zeros([rows, cols])
    U = np.zeros([rows, cols])
    for row_index in range(rows):
        for col_index in range(cols):
            if row_index > col_index:
                # when the row index is greater than the column index, it belongs in the L matrix
                L[row_index, col_index] = A[row_index, col_index]
            elif row_index < col_index:
                # when the row index is less than the column index, it belongs in the U matrix
                U[row_index, col_index] = A[row_index, col_index]
            else:
                # this will get all of the diagonal components of A and put them into D
                D[row_index, col_index] = A[row_index, col_index]
                D_inv[row_index, col_index] = A[row_index, col_index]**(-1)

    iter1 = b - np.dot(L+U, guess)
    iter1 = np.dot(D_inv, iter1)
    iter_history = {0: guess,
                    1: iter1}

    t = tolerance+1
    k = 1
    while t > tolerance:
        next_iter = b - np.dot(L+U, iter_history.get(k))
        next_iter = np.dot(D_inv, next_iter)
        # print(f"iteration ({k+1}): {next_iter}")
        iter_history.update({k+1: next_iter})

        t = np.linalg.norm(next_iter - iter_history.get(k)) / np.linalg.norm(iter_history.get(k))
        k += 1
    return k, next_iter

def gauss_seidel(A, b, guess, sor=1, tolerance=1e-8):
    # making sure that A is an nxn matrix and b is a 1xn vector
    try:
        np.dot(A,b)
    except:
        raise Exception("Make sure that A is an nxn matrix and that b is a vector with n elements")
    A = A.astype(float)
    b = b.astype(float)
    guess = guess.astype(float)
    rows, cols = np.shape(A)

    iter_history = {0: guess}
    # sol1 = (1-sor)*iter_history.get(0)[0]
    # sol1 += sor/A[0, 0] * (b[0] - A[])
    next_iter = iter_history.get(0).copy()
    for i in range(rows):
        
        sol_i = (1-sor)*iter_history.get(0)[i] # SOR only
        sol_i += sor/A[i, i] * b[i]
        
        for j in range(cols):
            if j == i:
                pass
            else:
                sol_i -= sor/A[i, i] * A[i, j] * next_iter[j]
        next_iter[i] = sol_i
        # print(next_iter)
    iter_history.update({1: next_iter})

    t = tolerance + 1 # calculated tolerance
    k = 1
    while t > tolerance:
        next_iter = iter_history.get(k).copy()
        for i in range(rows):
            sol_i = (1-sor)*iter_history.get(k)[i] # SOR only
            sol_i += sor/A[i, i] * b[i]
            for j in range(cols):
                if j == i:
                    pass
                else:
                    sol_i -= sor/A[i, i] * A[i, j] * next_iter[j]
            next_iter[i] = sol_i
            # print(next_iter)

        t = np.linalg.norm(next_iter - iter_history.get(k)) / np.linalg.norm(iter_history.get(k))
        k += 1    
        iter_history.update({k: next_iter})
        # print(t)
        # print(f"After {k} iterations, we have a solution {next_iter}")
    return k, next_iter

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
    guess = np.zeros(5)

    # print(gaussian_elimination(A,b))

    iterations, solution = jacobi_iteration(A, b, guess) # tolerance = 1e-08
    print(f"For Jacobi Iteration, it took {iterations} iterations to reach a solution of {solution}")
    print()
    iterations, solution = jacobi_iteration(A, b, guess, tolerance=1e-06) # tolerance = 1e-06
    print(f"For Jacobi Iteration, it took {iterations} iterations to reach a solution of {solution}")
    print()

    iterations, solution = gauss_seidel(A, b, guess) # tolerance = 1e-08
    print(f"For Gauss-Seidel, it took {iterations} iterations to reach a solution of {solution}")
    print()
    iterations, solution = gauss_seidel(A, b, guess, tolerance=1e-06) # tolerance = 1e-06
    print(f"For Gauss-Seidel, it took {iterations} iterations to reach a solution of {solution}")
    print()

    iterations, solution = gauss_seidel(A, b, guess, sor=1.1) # tolerance = 1e-08
    print(f"For SOR, it took {iterations} iterations to reach a solution of {solution}")
    print()
    iterations, solution = gauss_seidel(A, b, guess, sor=1.1, tolerance=1e-06) # tolerance = 1e-06
    print(f"For SOR, it took {iterations} iterations to reach a solution of {solution}")
    A0 = np.array([[3, -1, 1],
                   [3, 6, 2],
                   [3, 3, 7]])
    b0 = np.array([1, 0, 4])
    guess0 = np.array([0, 0, 0])
    gauss_seidel(A0, b0, guess0)