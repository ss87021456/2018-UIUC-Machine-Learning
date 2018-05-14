"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers


def train_model(data, model, learning_rate=0.001, batch_size=32,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def ceil(a, b):
        return a//b + bool(a % b)

    # ceiling = int(math.ceil(x.shape[0] / batch_size))
    x = data['image']
    y = data['label']
    ceiling = int(ceil(x.shape[0], batch_size))

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
        forward = model.forward(x_batch)
        grad = model.backward(forward, y_batch)
        model.w = model.w - (learning_rate * grad)

        if (iteration) % 100 == 0:
            loss = model.total_loss(forward, y_batch)
            predict = model.predict(forward)
            accuracy = np.sum(
                predict[:, np.newaxis] == y_batch)*100/len(predict)
            print("step {:3d} / {:3d} loss:{:3.2f} \
                accuracy:{:3.2f}%".format(
                    iteration, num_steps, loss, accuracy))

        # if iteration == num_steps-1:
        # predict = model.predict(forward)
        # for i, j in zip(predict[:, np.newaxis], y_batch):
        # print("prediction:{0}\t| GT:{1}".format(i[0], j[0]))

    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    pass


def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    # print (P.shape, q.shape, G.shape, h.shape)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])
    # Your implementation here (do not modify the code above)
    sv = z > 1e-5
    sv_index = np.arange(len(z))[sv.flatten()]
    alpha = z[sv_index]

    sv_X = data['image'][sv_index]
    sv_y = data['label'][sv_index]

    sv_X_ = np.concatenate((sv_X, np.ones((sv_X.shape[0], 1))), axis=1)
    # Set model.w
    model.w = np.zeros((sv_X_.shape[1], 1))
    for n in range(len(alpha)):
        model.w += (alpha[n] * sv_X_[n] * sv_y[n])[:, np.newaxis]


def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    C = 100  # assume C = 100
    x_ = data['image']
    x = np.concatenate((x_, np.ones((x_.shape[0], 1))), axis=1)

    n_samples = len(x)

    K = x.dot(x.T)
    y = data['label'].flatten()

    P = np.outer(y, y) * K
    q = np.ones((n_samples, 1)) * -1

    tmp1 = np.diag(np.ones(n_samples) * -1)
    tmp2 = np.identity(n_samples)
    G = np.vstack((tmp1, tmp2))

    tmp1 = np.zeros((n_samples, 1))
    tmp2 = np.ones((n_samples, 1)) * C
    h = np.vstack((tmp1, tmp2))

    return P, q, G, h


def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    f = model.forward(data['image'])
    loss = model.total_loss(f, data['label'])

    predict = model.predict(f)
    acc = np.sum(predict[:, np.newaxis] == data['label']) / len(predict)
    return loss, acc
