
# K-NN implementation
# Author: Andrew Bevilacqua
# CS-422 Project 1

import math
import numpy as np
def calc_euclidean_dist(x,y):
    dist = 0
    for i in range(len(x) - 1):
        dist += (x[i] - y[i]) ** 2
    return math.sqrt(dist)


def get_neighbors(X_train,Y_train,X_test,Y_test,K):
    dist_L = list()
    a = 0
    for training in X_train:
        dist = calc_euclidean_dist(training, X_test[a])
        dist_L.append((dist,training))
        a += 1

def KNN_test(X_train,Y_train,X_test,Y_test,K):
    X_test = X_test.astype(int)
    X_train = X_train.astype(int)
    print(X_train)
    get_neighbors(X_train,Y_train,X_test,Y_test,K)
    return 0



