# Ethan Watanabe
# Homework 3
# Question 3

import numpy as np
import matplotlib.pyplot as plt
from matrix_solvers import gaussian_elimination

def least_squares(x, y, degree=1):
    """
    Least Squares regression coefficient generator

    Takes data points x and data points y and generates coefficients for an n-degree best fit line

    Args:
        x (numpy.ndarray): an array containing independent variable data points
        y (numpy.ndarray): an array containing dependent variable data points
        degree (int): degree of polynomial for best fit (defaulted at 1 for linear regression)

    Returns:
        coefs (numpy.ndarray): an array containg the coefficients for best fit function
    """
    x = x.astype(float)
    y = y.astype(float)
    rows = len(x)

    A = np.ones([rows, degree + 1])

    for k in range(degree):
        for i in range(rows):
            A[i, k+1] = x[i] ** (k+1)
    A_t = np.matrix_transpose(A)

    coefs, aug_M = gaussian_elimination(np.dot(A_t, A), np.dot(A_t, y))
    return coefs

if __name__ == '__main__':
    temp_data = np.array([289.99, 300.12, 320.53, 346.13, 366.00, 369.87])
    coeff_data = -1 * np.array([5.576e-4, 1.039e-4, 2.255e-4, 2.285e-4, 2.894e-4, 2.902e-4])
    coefficients = least_squares(temp_data, coeff_data, degree=1)
    print(coefficients)

    temps = np.array([310, 330, 360])
    temp_coeffs = coefficients[0] + coefficients[1] * temps
    for n in range(len(temps)):
        print(f"The temperature coefficient of reactivity at {temps[n]}K is {temp_coeffs[n]}")
        print()