#

import numpy as np
import math

# New file started 09/23/21 @ 20:28

def Reshape_array(X):
    a = np.array(X).reshape(len(X), len(X[0]))
    return a


def Calc_entropy(a,b,Y):
    if a == 0:
        a = 1
    elif b == 0:
        b = 1
    print("Calc_entropy(a): " + str(a))
    print("Calc_entropy(b): " + str(b))
    samp_len = len(Y)
    H = -((a / samp_len) * math.log((a / samp_len), 2)) - ((b / samp_len) * math.log((b / samp_len), 2))
    return H


def Calc_total_entropy(Y):
    rght = 0
    lft = 0
    for L in Y:
        if L == 1:
            rght += 1
        else:
            lft += 1
    total_H = Calc_entropy(rght, lft, Y)
    return total_H

def Leaf_accuracy(leaf,Y):
    a = 0
    rght = 0
    wrng = 0
    while a < len(leaf):
        if Y[leaf[a]] == 1 or Y[leaf[a]] == '1':
            rght += 1
        else:
            wrng += 1
        a += 1
    return rght, wrng

def Calc_information_gain(right_arr,left_arr,Y):
    rght_r, lft_r = Leaf_accuracy(right_arr, Y)  # calc the accuracy on the right side of tree
    print("Calc_infoG(rightAcc)" + str(rght_r))
    print("Calc_infoG(rightAcc)" + str(lft_r))
    rght_l, lft_l = Leaf_accuracy(left_arr, Y)  # calc the accuracy on the left side of tree
    total_H = Calc_total_entropy(Y)  # = H()
    right_H = Calc_entropy(rght_r, lft_r, Y)  # = H(f=1)
    left_H = Calc_entropy(rght_l, lft_l, Y)  # = H(H!=1)
    print("total: " + str(total_H))
    print("right: " + str(right_H))
    print("left: " + str(left_H))
    IG = total_H - right_H - left_H  # IG = H() - H(f=1) - H(f!=1)
    return IG

def DT_train_binary_rec(r,c,X,Y,IG):
    if r < len(Y):  # if row number is less than the total length of labels/samples
        left_n = []
        right_y = []
        if int(X[r][c]) == 1:  # checks if feature of each row is = 1
            right_y.append(r)  # yes query is true
        else:
            left_n.append(r)  # no query is false
        print("R " + str(r) + "C " + str(c) +"  R_y: " + str(right_y) + "L_n: " + str(left_n))
    else:
        r = 0
        c += 1
        print(len(Y))
    r += 1
    DT_train_binary_rec(r, c, X, Y, IG)



def DT_train_binary(X,Y,max_depth):
    IG = []
    if len(X) != len(Y):  # Checks if the rows of X are equal to the columns of Y
        print("Error: Rx != Cy")
        return 0
    elif len(X) == 0 or len(Y) == 0:  # checks too see if either X or Y is empty
        print("Error_2:Empty List")
    else:
        DT_train_binary_rec(0,0, X, Y, IG)
        print("IG: " + str(IG))
    return 0


def DT_test_binary(X,Y,DT):
    return 0


def DT_make_prediction(x,DT):
    return 0


def DT_train_real(X,Y,max_depth):
    return 0


def DT_test_real(X,Y,max_depth):
    return 0