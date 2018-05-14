from copy import deepcopy
import numpy as np
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset
data = pd.read_csv('./data/data/iris.data')

# Make 3  clusters
k = 3
# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
C = np.array(C)
print("Initial Centers")
print(C)

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def k_means(C):
    # Write your code here!
    global data
    global k
    y = data['V5']
    X = data.drop(['V5'],axis=1)
    X = X.as_matrix()   # convert to numpy array
    error = 1000        # initial error
    clusters = np.zeros(len(X))
    C = np.array(C)     # in case input list type

    while error > 10e-3:
        C_old = C.copy()

        # Determine new cluster labels
        for i in range(len(X)):
            distances = dist(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster

        # Determine new cluster points
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)

        error = dist(C, C_old, None)
        print("error: {:3f}".format(error))

    # print(y.head(), clusters[:5])

    return C








