"""logistic model class for binary classification."""

import numpy as np


class LogisticModel(object):

    def __init__(self, ndims=16, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of W is the bias term,
            self.W = [Bias, W1, W2, W3, ...]
            where Wi correspnds to each feature dimension

        W_init needs to support:
          'zeros': initialize self.W with all zeros.
          'ones': initialze self.W with all ones.
          'uniform': initialize self.W with uniform random number between [0,1)
          'gaussian': initialize self.W with gaussion distribution (0, 0.1)

        Args:
            ndims(int): feature dimension
            W_init(str): types of initialization.
        """
        self.ndims = ndims
        self.W_init = W_init
        self.W = None
        ###############################################################
        # Fill your code below
        ###############################################################
        weight_shape = ndims+1

        if W_init == 'zeros':
            self.W = np.zeros(weight_shape)
        elif W_init == 'ones':
            self.W = np.ones(weight_shape)
        elif W_init == 'uniform':
            self.W = np.random.rand(weight_shape)
        elif W_init == 'gaussian':
            mu, sigma = 0, 0.1
            self.W = np.random.normal(mu, sigma, weight_shape)
        else:
            print('Unknown W_init ', W_init)

    def save_model(self, weight_file):
        """ Save well-trained weight into a binary file.
        Args:
            weight_file(str): binary file to save into.
        """
        self.W.astype('float32').tofile(weight_file)
        print('model saved to', weight_file)

    def load_model(self, weight_file):
        """ Load pretrained weghit from a binary file.
        Args:
            weight_file(str): binary file to load from.
        """
        self.W = np.fromfile(weight_file, dtype=np.float32)
        print('model loaded from', weight_file)

    def forward(self, X):
        """ Forward operation for logistic models.
            Performs the forward operation, and return probability
            score (sigmoid).
        Args:
            X(numpy.ndarray): input dataset with a dimension of (#
            of samples, ndims+1)
        Returns:
            (numpy.ndarray): probability score of (label == +1) for
            each sample
                             with a dimension of (# of samples,)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        W_ = self.W[:, np.newaxis]
        w_t_x = np.dot(X, W_)

        # sigmoid
        score = 1 / (1 + np.exp(-w_t_x))

        return score.flatten()

    def backward(self, Y_true, X):
        """ Backward operation for logistic models.
            Compute gradient according to the probability loss on
            lecture slides
        Args:
            X(numpy.ndarray): input dataset with a dimension of (#
            of samples, ndims+1)
            Y_true(numpy.ndarray): dataset labels with a dimension
            of (# of samples,)
        Returns:
            (numpy.ndarray): gradients of self.W
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        Y_ = Y_true[:, np.newaxis]
        W_ = self.W[:, np.newaxis]
        nomi = -Y_ * X * np.exp(-Y_ * np.dot(X, W_))
        # print(nomi.shape)
        domi = 1 + np.exp(-Y_ * np.dot(X, W_))
        # print(domi.shape)
        grad = np.sum(nomi/domi, axis=0)
        # print(grad.shape)
        # exit()
        return grad

    def classify(self, X):
        """ Performs binary classification on input dataset.
        Args:
            X(numpy.ndarray): input dataset with a dimension of (#
            of samples, ndims+1)
        Returns:
            (numpy.ndarray): predicted label = +1/-1 for each sample
                             with a dimension of (# of samples,)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        prediction = self.forward(X)
        result = [1 if round(pre) else -1 for pre in prediction]
        result = np.asarray(result)
        return result

    def fit(self, Y_true, X, learn_rate, max_iters, batch_size=32):
        """ train model with input dataset using gradient descent.
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of
            (# of samples,)
            X(numpy.ndarray): input dataset with a dimension of
            (# of samples, ndims+1)
            learn_rate: learning rate for gradient descent
            max_iters: maximal number of iterations
            ......: append as many arguments as you want
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        def unison_shuffled_copies(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        def ceil(a, b):
            return a//b + bool(a % b)
        ceiling = int(ceil(X.shape[0], batch_size))
        # ceiling = int(math.ceil(x.shape[0] / batch_size))
        x = X

        for iteration in range(max_iters):
            if True and iteration % ceiling == 0:
                x, Y_true = unison_shuffled_copies(x, Y_true)
            batch_begin = (iteration % ceiling) * batch_size
            batch_end = ((iteration+1) % ceiling) * batch_size

            # meet end of each epoch
            if ((iteration+1) % ceiling) == 0 and iteration != 0:
                x_batch, y_batch = x[batch_begin:], Y_true[batch_begin:]
            else:
                x_batch, y_batch = x[batch_begin: batch_end], \
                                   Y_true[batch_begin: batch_end]

            # Gradient descent perfom here
            grad = self.backward(y_batch, x_batch)
            self.W = self.W - (learn_rate * grad)

            if (iteration) % 100 == 0:
                loss = np.sum(np.log(1 + np.exp(
                    -y_batch * np.dot(x_batch, self.W))))
                predict = self.classify(x_batch)
                accuracy = np.sum(predict == y_batch) * 100 / len(predict)
                print("step {:3d} / {:3d} loss:{:3.2f} \
                    accuracy:{:3.2f}%".format(
                        iteration, max_iters, loss, accuracy))

            if iteration == max_iters-1:
                predict = self.classify(x_batch)
                # for i, j in zip(predict, y_batch):
                # print("prediction:{0}\t| GT:{1}".format(i, j))
        # Perform gradient descent.
        pass
