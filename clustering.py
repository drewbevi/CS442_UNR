
# K_means implementation
# Author: Andrew Bevilacqua
# CS-422 Project 1
import math
import numpy as np
import random as rnd
def calc_dist(x,y):
    a = abs(x - y)
    return float(a)

def calc_euclidean_dist(x,y):
    dist = 0
    for i in range(len(x) - 1):
        dist += (x[i] - y[i]) ** 2
    a = math.sqrt(dist)
    return a

def calc_mean(K,mu,count):
    x = 0
    np_temp = np.zeros(shape=K)
    mu_new = np_temp.astype(int)
    while x < K:
        mu_new = np.append(mu_new,[mu[x] / count[x]], axis=0)
        x += 1
    return mu_new[2:]

def calc_mean2(K,mu,count):
    x = 0
    np_temp = np.zeros(shape=(K, len(mu[0])))
    mu_new = np_temp.astype(int)
    while x < K:
        mu_new = np.append(mu_new,[mu[x] / count[x]], axis=0)
        x += 1
    print(mu_new)
    return mu_new

def rnd_mu(X,K,mu):
    np_temp = np.zeros(shape=(K, len(X[0])))
    mu_new = np_temp.astype(int)
    x = 0
    while x < K:
        mu_new[x] = X[rnd.randrange(0, (len(X)-1))]
        x += 1
    return mu_new

def rnd_K(X,K,mu):
    return 0

def find_min(x,y):
    if x < y:
        return x
    elif x > y:
        return y
    else:
        return x or y

def K_mean_rec(index_X, index_mu, mu_count, cluster, X, K, mu):
    if index_X < (len(X)):
        mu_dist_temp = float(0)
        mu2_dist_temp = float(0)
        mu_dist_temp = calc_dist(X[index_X], mu[index_mu])
        mu2_dist_temp = calc_dist(X[index_X], mu[index_mu + 1])
        mu_min = min(mu_dist_temp, mu2_dist_temp)  # calc min dist between points and mean

        if mu_min == mu_dist_temp:  # if the min dist is that of the first mean
            mu_count[index_mu] += 1  # increase first mean count
            cluster[index_mu] += X[index_X]  # add data point to temp mu

        elif mu_min == mu2_dist_temp:  # if the min dist is that of the second mean
            mu_count[index_mu + 1] += 1  # increase second mean count
            cluster[index_mu + 1] += X[index_X]  # add data point to temp mu
        K_mean_rec(index_X + 1, index_mu, mu_count, X, mu, cluster, K)
    else:
        mu_temp = calc_mean(K,cluster, mu_count)  # calc new means return new means

        if mu[index_mu] == mu_temp[index_mu] and mu[index_mu + 1] == mu_temp[index_mu + 1]:  # checks if current mu = new mu after iteration
            print("KMeans(unable to return value): " + str(mu_temp))
            return mu_temp  # cannot seem to return this value
        else:  # if not then do new iteration
            K_mean_rec(0, index_mu, mu_count, X, mu_temp, cluster, K)

def K_mean_rec2(index_X, index_mu, mu_count, cluster1, cluster2, X, K, mu):
    if index_X < (len(X)):
        mu_dist_temp = float(0)
        mu2_dist_temp = float(0)
        mu_dist_temp += calc_euclidean_dist(X[index_X], mu[index_mu])
        mu2_dist_temp += calc_euclidean_dist(X[index_X], mu[index_mu + 1])
        mu_min = min(mu_dist_temp, mu2_dist_temp)  # calc min dist between points and mean

        if mu_min == mu_dist_temp:  # if the min dist is that of the first mean
            cluster1 = np.append(cluster1,X[index_X])
        elif mu_min == mu2_dist_temp:  # if the min dist is that of the second mean
            cluster2[index_mu] = X[index_X]
        K_mean_rec2(index_X+1, index_mu, mu_count, cluster1, cluster2, X, K, mu)

    else:
        mu_temp = np.mean(cluster1)  # calc new means return new means
        np_temp = np.zeros(shape=(len(X), K))
        if np.array_equal(mu[index_mu], mu_temp[index_mu]) and np.array_equal(mu[index_mu + 1], mu_temp[index_mu + 1]):  # checks if current mu = new mu after iteration
            print("KMeans(unable to return value): " + str(mu_temp))
            return mu_temp  # cannot seem to return this value
        else:  # if not then do new iteration
            K_mean_rec2(0, index_mu, np_temp.astype(int), np_temp.astype(int),np_temp.astype(int), X, K, mu)

def K_Means(X,K,mu):

    if K == 0:  # if no specified K find a random one
        print("No K was given therefore no clusters were created")
        return 0
    elif mu == [[]]:  # random mean values must be assigned
        mu = rnd_mu(X,K,mu)
    elif X == [[]]:  # if the training feature data is empty
        print("Error: empty training feature data")
        return 0
    X = np.int_(X)  # turns elems in X from str to int
    np_temp = np.zeros(shape=(len(X[0]), K))
    count = np_temp.astype(int)
    np_temp = np.zeros(shape=(len(X), K))
    cluster = np_temp.astype(int)
    if len(X[0]) > 1:
        print("Multi-dimensional k_means = WIP")
        # return K_mean_rec2(0, 0, count, cluster, cluster, X, K, mu)
    else:
        return K_mean_rec(0, 0, count, cluster, X, K, mu)

