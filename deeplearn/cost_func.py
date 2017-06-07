"""
Cost functions.
"""

import numpy as np
from numba import jit

from .funcs import sigmoid


class Cost(object):
    """Cost function meta class.
    """


class BernSig(Cost):
    """
    The cost function for a bernoulli distribution parametrized by the Sigmoid
    activation function.
    """
    def __init__(self):
        pass

    @jit(nogil=True)
    def forward(self, P, Y):
        """Calculate the Euclidean norm.

        Args
            Y (array): Left matrix for difference
            P (array): Right matrix for difference

        Returns
            n (scalar): (1/2)|| Y - P ||^2.
        """
        Y = Y.ravel()
        P = P.ravel()
        H = np.multiply(1 - 2 * Y, P)
        return np.mean(np.log(1 + np.exp(H)))

    @jit(nogil=True)
    def backprop(self, P, Y, H, G):
        """Gradient of the Norm.

        Args
        Y (array): Left matrix of norm calculation
        X (array): Right matrix of norm calculation
        H (None): null argument for compatibility
        G (array): null argument for compatbility

        Returns
        dNorm (list): [A, B], where A is the gradient of the norm and B is
            a null argument for compatibility.
        """
        n = Y.shape[0]
        Y = Y.reshape(n, 1)
        H = np.multiply(1 - 2 * Y, P)
        return [(1 / n) * sigmoid(H), None]


class Norm(Cost):

    """
    Gate for Euclidean Norm.
    """

    def __init__(self):
        pass

    @jit(nogil=True)
    def forward(self, P, Y):
        """Calculate the Euclidean norm.

        Args
            Y (array): Left matrix for difference
            P (array): Right matrix for difference

        Returns
            n (scalar): (1/2)|| Y - P ||^2.
        """
        Y = Y.ravel()
        P = P.ravel()
        H = Y - P

        C = H.dot(H)
        return (1 / (2 * Y.shape[0])) * C

    @jit(nogil=True)
    def backprop(self, P, Y, H, G):
        """Gradient of the Norm.

        Args
        Y (array): Left matrix of norm calculation
        X (array): Right matrix of norm calculation
        H (None): null argument for compatibility
        G (array): null argument for compatbility

        Returns
        dNorm (list): [A, B], where A is the gradient of the norm and B is
            a null argument for compatibility.
        """
        n = Y.shape[0]
        H = Y.reshape(n, 1) - P
        return [- (1 / n) * H, None]
