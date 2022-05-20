# Neural Network implementation
# Author: Andrew Bevilacqua
# CS-422 Project 3

import numpy as np
import random as rnd

def sum_list(L):
    sum = 0
    for items in L:
        sum += items
    return sum
def tanh(x, b, deriv = False):
    t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) + b
    if deriv:
        return 1- t * t
    return t
def softmax(z):
    e = np.exp(z)
    return e / e.sum()
def softmax_dervi(s):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

def calculate_loss(model, X, y, y_hat):
    loss = -1/len(X) * sum(y * np.log(y_hat))
    return loss

def predict(model, x):
    h = tanh(np.dot(model.get("W1"),x))
    return np.dot(model.get("W2"),h)

def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    model = {"W1":np.tile(rnd.random(),(len(X[0]),nn_hdim)), "b1":0, "W2":np.tile(rnd.random(),(len(X[0]),nn_hdim)), "b2":0}
    for iter in range(2):
        G = np.tile(0, (len(X[0]), nn_hdim))  # gradient variable for hidden layer
        g = np.tile(0, nn_hdim)  # gradient variable for output layer
        index = 0
        for x in X:
            for i in range(nn_hdim):
                h_2D = tanh(x, model["b1"])  # calcs activation for hidden layer nodes
                h = sum_list(h_2D)
            z = np.add(np.dot(model["W2"], h), model["b2"])  # calcs activation for output layer nodes
            y_hat = softmax(z)
            loss = calculate_loss(model, X, y[index],y_hat )
            g = g - loss * h  # the gradient for the output layer is updated
            for i in range(nn_hdim):
                G[i] = G[i] - loss * model["W2"][i](tanh(h, model["b2"], True)) * x  # the gradient for the hidden layer is updated
            index += 1
        model["W1"] += -1 * 0.01 * G  # weights are updated
        model["W2"] += -1 * 0.01 * g
    return model

