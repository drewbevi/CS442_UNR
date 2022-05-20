# Perceptron implementation
# Author: Andrew Bevilacqua
# CS-422 Project 2
import numpy as np


# Perceptron Train Functions
def calc_activation(x,w,b,):
    a = 0
    for i in range(len(x) - 1):  # iterates through feature data of sample
        a += w[i] * x[i]  # calculates activation without bias
    a += b  # add bias
    return a

def predict(x,y,w,b):
    a = calc_activation(x,w,b)  # calls on function to find activation
    if np.all(a * y <= 0):  # checks if the activation is less than or equal to zero
        w,b = update_vals(x,y,w,b)  # if yes then w and b are updated
        return w,b
    else:  # if no then w and b are returned
        return w,b

def update_vals(x,y,w,b):
    w += y * x  # w <- w + x * y
    b += y  # b <- b + y
    return w,b

def perceptron_train(X, Y):
    w = np.zeros(len(X[0]))  # weight, size of a single sample
    b = 0  # bias variable
    max_epoch = 10  # max epoch value
    for e in range(max_epoch):  # iterates through the max epochs
        for i in range(len(X-1)):  # iterates through samples
            w, b = predict(X[i],Y[i],w,b)  # calls predict function to calc w and b
    return w,b


# Perceptron Test Functions
def predict_test(x,y,w,b):
    a = 0  # declares activation to be 0
    for i in range(len(x)-1):  # iterates through feature data of sample
        a += w[i + 1] * x[i]  # calculates activation without bias
    a += b  # add bias
    return 1.0 if y*a > 0.0 else -1.0  # return 1 if prediction is correct, -1 if not

def perceptron_test(X_test, Y_test, w, b):
    c = 0  # variable used to keep count of accurate predication
    y = 0  # index variable
    for x in X_test:  # iterates through test data
        c += predict_test(x,Y_test[y],w,b)  # calculates the number of correct predictions
        y+=1
    acc = c/len(X_test-1)  # calculates accuracy of predictions
    return abs(acc)

