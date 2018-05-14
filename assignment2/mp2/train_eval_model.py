"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import math
from models.linear_regression import LinearRegression

x_max = None
x_min = None
x_rng = None


def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    global x_max, x_min, x_rng

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def scale_linear_bycolumn(rawpoints, high=1.0, low=0.0):
        mins = np.min(rawpoints, axis=0)
        maxs = np.max(rawpoints, axis=0)
        rng = maxs - mins
        return high - (((high - low) * (maxs - rawpoints)) / rng)

    x = processed_dataset[0]
    y = processed_dataset[1]

    # Perform Min-Max Scale
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    x_rng = x_max - x_min
    x = 1.0 - (((1.0 - 0.0) * (x_max - x)) / x_rng)

    ceiling = int(math.ceil(x.shape[0] / batch_size))

    for iteration in range(num_steps):
        if shuffle and iteration % ceiling == 0:
            x, y = unison_shuffled_copies(x, y)
        batch_begin = (iteration % ceiling) * batch_size
        batch_end = ((iteration+1) % ceiling) * batch_size

        # meet end of each epoch
        if ((iteration+1) % ceiling) == 0 and iteration != 0:
            x_batch, y_batch = x[batch_begin:], y[batch_begin:]
        else:
            x_batch, y_batch = x[batch_begin: batch_end], \
                               y[batch_begin: batch_end]

        # Gradient descent perfom here
        update_step(x_batch, y_batch, model, learning_rate)

        if (iteration+1) % 100 == 0:
            loss = model.total_loss(model.forward(x_batch), y_batch)
            print("iteration {0} loss:{1}".format(iteration,loss))

        if iteration == num_steps-1:
            predict = model.forward(x_batch)
            for i, j in zip(predict, y_batch):
                print("prediction",i,"| GT",j)

    # Perform gradient descent.

    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initiali x_min = np.min(x, axis=0)zed linear model.
    """
    forward = model.forward(x_batch)
    backward = model.backward(forward, y_batch) / len(x_batch)
    model.w += -learning_rate * backward


def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    global x_max, x_min, x_rng
    x = processed_dataset[0]
    y = processed_dataset[1]
    # Min-Max Scale
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    x_rng = x_max - x_min
    x = 1.0 - (((1.0 - 0.0) * (x_max - x)) / x_rng)
    X = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)

    X_transpose = np.matrix.transpose(X)
    Identity = np.identity(len(X[1, :]))
    Identity[0, 0] = 0

    BetaHat = np.dot(np.linalg.pinv(np.add(np.dot(X_transpose, X),
                     model.w_decay_factor*Identity)), np.dot(X_transpose, y))

    model.w = BetaHat
    # print("BetaHat shape:",BetaHat.shape)
    # print(BetaHat)

    return model


def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    x = processed_dataset[0]
    y = processed_dataset[1]

    # perform Min-Max Scale
    x = 1.0 - (((1.0 - 0.0) * (x_max - x)) / x_rng)

    loss = model.total_loss(model.forward(x), y)
    predict = model.forward(x[:10])
    # for i, j in zip(predict, y[:10]):
    #    print("prediction",int(i[0]),"\t | GT",j[0])

    return loss
