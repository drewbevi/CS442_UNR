import numpy as np
import random as rnd

def calc_activation(x,w,b):
    a = 0
    length = len(x)
    for i in range(length):  # iterates through feature data of sample
        a += w[i] * x[i]  # calculates activation without bias
    a += b  # add bias
    return a[0]

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return t

def update_input_gradient(G, error, model, a, x):
    G = G - error * model.get("W2") * (1 - (tanh(a) * tanh(a))) * x [0]

def calculate_loss(model, X, y):
    return 0

def predict(model, x):
    h = tanh(np.dot(model.get("W1"),x))
    return np.dot(model.get("W2"),h)


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    model = {"W1":[rnd.random(), rnd.random()], "b1":[0], "W2":[rnd.random(), rnd.random()], "b2":[0]}
    for i in range(100):
        G = [rnd.random(), rnd.random()]
        g = [rnd.random(), rnd.random()]
        index = 0
        for x in X:
            for h_i in nn_hdim:
                a = np.dot(model.get("W1"), x) + model.get("b1")
                h = tanh(a)
            y_hat = np.dot(model.get("W2"),h)
            y_hat += model.get("b2")
            error = y[index] - y_hat
            g = g - error*h
            G = G - (error*model.get("W2")) * (1-(tanh(a)*tanh(a))) * x
            index += 1
    w1 = {"W1": [model["W1"][0] - 0.01*G[0], model["W1"][1] - 0.01*G[1]]}
    w2 = {"W2": [model["W2"][0] - 0.01*G[0], model["W2"][1] - 0.01*G[1]]}
    model.update(w1)
    model.update(w2)
    print(model)
    return 0
