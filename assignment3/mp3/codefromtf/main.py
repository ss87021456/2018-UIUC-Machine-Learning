"""Main function for binary classifier"""
import tensorflow as tf
import numpy as np
from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 1e-2
max_iters = 5000


def main(_):
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    # Load dataset.
    A, T = read_dataset_tf('../data/trainset', 'indexing.txt')
    A, T = unison_shuffled_copies(A, T)
    # Hint: A, T = read_dataset_tf('../data/trainset','indexing.txt')
    # Initialize model.
    model = LogisticModel_TF(ndims=16, W_init='uniform')
    # Build TensorFlow training graph
    model.build_graph(learn_rate=learn_rate)
    # Train model via gradient descent.
    result = model.fit(T, A, max_iters, batch_size=16)
    # Compute classification accuracy based on the return of the "fit" method
    # print(np.round(result)[:20])
    # print(T[:20])
    accuracy = np.sum(np.round(result) == T) * 100 / len(result)
    print("overall accuracy : {:3.2f}% {:2d} / {:2d}".format(
        accuracy, np.sum(np.round(result) == T), len(result)))


if __name__ == '__main__':
    tf.app.run()
