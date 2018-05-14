"""Main function for binary classifier"""

import numpy as np

from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 1e-4
max_iters = 10000

if __name__ == '__main__':
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset('../data/trainset','indexing.txt')
    A, T = read_dataset('../data/trainset', 'indexing.txt')
    # Initialize model.
    model = LogisticModel(ndims=16, W_init='ones')
    # Train model via gradient descent.
    model.fit(T, A, learn_rate, max_iters, batch_size=32)
    # Save trained model to 'trained_weights.np'
    model.save_model('trained_weights.np')
    # Load trained model from 'trained_weights.np'
    model.load_model('trained_weights.np')
    # Try all other methods: forward, backward, classify, compute accuracy
    predict = model.classify(A)
    accuracy = np.sum(predict == T) * 100 / len(predict)
    print("testing accuracy:{:3.2f}%".format(accuracy))
    # print(model.forward(A).shape)
    # print(model.backward(T, A).shape)
