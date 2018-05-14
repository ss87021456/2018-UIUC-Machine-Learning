"""logistic model class for binary classification."""
import tensorflow as tf
import numpy as np


class LogisticModel_TF(object):

    def __init__(self, ndims=16, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based
        on the method
        specified in W_init.

        We assume that the FIRST index of Weight is the bias term,
            Weight = [Bias, W1, W2, W3, ...]
            where Wi correspnds to each feature dimension

        W_init needs to support:
          'zeros': initialize self.W with all zeros.
          'ones': initialze self.W with all ones.
          'uniform': initialize self.W with uniform random
           number between [0,1)
          'gaussian': initialize self.W with gaussion
           distribution (0, 0.1)

        Args:
            ndims(int): feature dimension
            W_init(str): types of initialization.
        """
        self.ndims = ndims
        self.W_init = W_init
        self.W0 = None
        ###############################################################
        # Fill your code below
        ###############################################################
        if W_init == 'zeros':
            # Hint: self.W0 = tf.zeros([self.ndims+1,1])
            self.W0 = tf.zeros([self.ndims+1, 1], dtype=tf.float64)
        elif W_init == 'ones':
            self.W0 = tf.ones([self.ndims+1, 1], dtype=tf.float64)
        elif W_init == 'uniform':
            self.W0 = tf.random_uniform([self.ndims+1, 1], dtype=tf.float64)
        elif W_init == 'gaussian':
            mu, sigma = 0, 0.1
            self.W0 = tf.random_normal(
                shape=[self.ndims+1, 1], mean=mu, stddev=sigma,
                dtype=tf.float64)
        else:
            print('Unknown W_init ', W_init)

    def build_graph(self, learn_rate):
        """ build tensorflow training graph for logistic model.
        Args:
            learn_rate: learn rate for gradient descent
            ......: append as many arguments as you want
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        # Hint: self.W = tf.Variable(self.W0)
        # tf Graph Input
        self.x = tf.placeholder(tf.float64, [None, 17])  # mfc feature number
        self.y = tf.placeholder(tf.float64, [None, 1])
        # 0, 1 labels => 2 classes
        self.W = tf.Variable(self.W0)

        # Construct model
        self.pred = tf.sigmoid(tf.matmul(self.x, self.W))

        # Minimize error using quadratic loss
        self.cost = tf.nn.l2_loss(self.y - self.pred)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learn_rate).minimize(self.cost)

        # accuracy
        self.correct_prediction = tf.equal(
            tf.round(self.pred), tf.round(self.y))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32)) * 100

        # init
        self.init = tf.global_variables_initializer()

    def fit(self, Y_true, X, max_iters, batch_size=16):
        """ train model with input dataset using gradient descent.
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension
             of (# of samples,1)
            X(numpy.ndarray): input dataset with a dimension of
             (# of samples, ndims+1)
            max_iters: maximal number of training iterations
            ......: append as many arguments as you want
        Returns:
            (numpy.ndarray): sigmoid output from well trained
            logistic model, used for classification
                             with a dimension of (# of samples, 1)
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
        x = X

        with tf.Session() as sess:

            # Run the initializer
            sess.run(self.init)

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

                # optimize perfom here
                _, cost = sess.run(
                    [self.optimizer, self.cost],
                    feed_dict={self.x: x_batch, self.y: y_batch})

                if (iteration) % 500 == 0:
                    acc = sess.run(
                        self.accuracy,
                        feed_dict={self.x: x_batch, self.y: y_batch})
                    print("step {:3d} / {:3d} loss:{:3.2f}  \
                        acc:{:3.2f}% ".format(iteration, max_iters, cost, acc))

            result = sess.run(self.pred, feed_dict={self.x: X})
            # print("result shape:",result.shape)
            return result
