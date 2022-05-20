import numpy as np
import decision_trees as dt
import nearest_neighbors as nn
import clustering as kmeans

def load_data(name):
    f = open(name, 'r')
    ctr = 0
    y_str = ''
    x_str = ''
    for line in f:
        line = line.strip().split(';')
        if ctr == 0:
            x_str = line
        else:
            y_str = line
        ctr += 1
    f.close()
    X = []
    Y = []
    for item in x_str:
        temp = [x for x in item.split(',')]
        X.append(temp)
    if len(y_str) > 0:
        for item in y_str:
            temp = int(item)
            Y.append(temp)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

X,Y = load_data("data_1.txt")
max_depth = 3
DT = dt.DT_train_binary(X,Y,max_depth)
test_acc = dt.DT_test_binary(X,Y,DT)
print("DT:",test_acc)


