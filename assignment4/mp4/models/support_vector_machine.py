"""Implements support vector machine."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
    """Implements a linear regression mode model"""

    def backward(self, f, y):
        """Performs the backward operation.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,).
        """
        def one_func(x):
            if x < 1:
                return 1
            else:
                return 0

        # x_ shape (N, ndim+1)
        x_ = np.concatenate((self.x, np.ones((self.x.shape[0], 1))), axis=1)
        # one_term shape (N, 1)
        one_term = np.apply_along_axis(one_func, 1, y*f)[:, np.newaxis]
        # reg_grad shape (ndim+1, 1)
        reg_grad = (self.w_decay_factor) * self.w
        loss_grad = np.dot(x_.T, y*one_term)
        # total_grad shape (ndim+1, 1)
        total_grad = reg_grad - loss_grad

        return total_grad

    def total_loss(self, f, y):
        """The sum of the loss across batch examples + L2 regularization.
        Total loss is hinge_loss + w_decay_factor*l2_loss

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum hinge loss + reguarlization.
        """
        def hinge_function(x):
            if 1 - x > 0:
                return 1 - x
            else:
                return 0

        hinge_loss = np.sum(np.apply_along_axis(hinge_function, 1, y*f))
        l2_loss = np.sum(0.5 * self.w_decay_factor * np.linalg.norm(self.w)**2)

        total_loss = hinge_loss + l2_loss
        return total_loss

    def predict(self, f):
        """Converts score to prediction.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,). Tie break 0 to 1.0.
        """
        # forward > 0 then return 1 else return -1 class
        y_predict = [1 if predict > 0 else -1 for predict in f]
        y_predict = np.asarray(y_predict)

        return y_predict
