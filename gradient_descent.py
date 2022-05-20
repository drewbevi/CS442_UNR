# Gradient Descent implementation
# Author: Andrew Bevilacqua
# CS-422 Project 2
import numpy as np


def gradient_descent(F, x, n):
    precision = 0.00001  # min magnitude of the gradient
    c = 1  # variable to keep count
    while True:  # while loop continues until internal condition statement is met
        diff = - n * F(x)  # equation calculates the first half of the gradient descent update function
        if np.all(np.abs(diff) <= precision):  # condition statement used to see if the magnitude is less than the min magnitude
            break
        x += diff  # add the difference to the x variable
        c += 1
    return x


