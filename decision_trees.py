
# Decision_trees implementation
# Author: Andrew Bevilacqua
# CS-422 Project 1

import math
import numpy as np

def Calc_entropy(a,b):
    global H
    if a == 0:
        H = -b * math.log(b, 2)
    elif b == 0:
        H = -a * math.log(a, 2)
    else:
        H = -(a * math.log(a, 2)) - (b * math.log(b, 2))
    return H


def Calc_total_entropy(Y):
    rght = 0
    lft = 0
    for L in Y:
        if L == 1:
            rght += 1
        else:
            lft += 1
    total_H = Calc_entropy(rght/len(Y), lft/len(Y))
    return total_H


def Calc_information_gain(acc_r,acc_l,Y):
    total_H = Calc_total_entropy(Y)  # = H()
    right_H = Calc_entropy(acc_r[0][0], acc_r[0][1])  # = H(f=1)
    left_H = Calc_entropy(acc_l[0][0], acc_l[0][1])  # = H(H!=1)
    IG = total_H - right_H - left_H  # IG = H() - H(f=1) - H(f!=1)
    return IG

def Leaf_accuracy(leaf,Y):  # returns the accuracy of the decision per leaf
    if len(leaf) == 0:
        return 0
    else:
        rght = 0
        wrng = 0
        a = 0
        for x in leaf:
            if leaf[a] == Y[x]:
                rght += 1
            else:
                wrng += 1

        r = rght/len(leaf)
        w = wrng/len(leaf)
        return r, w  # returns a %

def DT_rec(c,X,Y,IG):
    if X == []:
        print("done")
        return 0
    else:
        a = 0
        right_y = list()  # list containing index of samples on right of tree
        left_n = list()  # list containing index of samples on left of tree
        acc_y = list()
        acc_n = list()
        for x in X:
            if x[a] == 1:  # if the feature value is 1
                right_y.append(a)
            else:
                left_n.append(a)
            a += 1
        x_cp = np.delete(X,0,1)  # remove first elm in feature data and save to a copy
        acc_y.append(Leaf_accuracy(right_y,Y))
        acc_n.append(Leaf_accuracy(left_n,Y))
        IG.append(abs(Calc_information_gain(acc_y,acc_n,Y)))
        c += 1
        DT_rec(c,x_cp,Y,IG)


def DT_train_binary(X,Y,max_depth):
    if len(X) != len(Y):  # Checks if the rows of X are equal to the columns of Y
        print("Error: Rx != Cy")
        return 0
    elif len(X) == 0 or len(Y) == 0:  # checks too see if either X or Y is empty
        print("Error_2:Empty List")
    else: # some rec function to create/train DT
        X = X.astype(int)  # transforms str to int
        IG = list()
        DT_rec(0,X,Y,IG)
        return 0


def DT_test_binary(X,Y,DT):
    return 0


def DT_make_prediction(x,DT):
    return 0
