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
    def forward(self, inputs):
        """Calculate the Euclidean norm.

        Args
            Y (array): Left matrix for difference
            P (array): Right matrix for difference

        Returns
            n (scalar): (1/2)|| Y - P ||^2.
        """
        P, Y = inputs
        if Y is None:
            return None

        Y = Y.ravel()
        P = P.ravel()
        V = np.multiply(1 - 2 * Y, P)
        return np.mean(np.log(1 + np.exp(V)))

    @jit(nogil=True)
    def backprop(self, inputs, H, G):
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
        P, Y = inputs
        n = Y.shape[0]
        Y = Y.reshape(n, 1)
        V = np.multiply(1 - 2 * Y, P)
        return [(1 / n) * sigmoid(V), None]


class Norm(Cost):

    """
    Gate for Euclidean Norm.
    """

    def __init__(self):
        pass

    @jit(nogil=True)
    def forward(self, inputs):
        """Calculate the Euclidean norm.

        Args
            Y (array): Left matrix for difference
            P (array): Right matrix for difference

        Returns
            n (scalar): (1/2)|| Y - P ||^2.
        """
        P, Y = inputs
        if Y is None:
            return None

        Y = Y.ravel()
        P = P.ravel()
        V = Y - P

        C = V.dot(V)
        return (1 / (2 * Y.shape[0])) * C

    @jit(nogil=True)
    def backprop(self, inputs, H, G):
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
        P, Y = inputs
        n = Y.shape[0]
        V = Y.reshape(n, 1) - P
        return [- (1 / n) * V, None]


class Softmax(Cost):

    """ Cross-entropy of the Softmax function.
    """

    def __init__(self):
        pass

    @jit(nogil=True)
    def forward(self, inputs):
        """Calculate the sigmoid element-wise in a forward pass.

        -log softmax(z)_{y=i} = -log ( exp(z_i) / sum exp(z_j)
                              = - z_i + log sum exp(z_j)

        Args
        inputs (tuple):
            X (array): Class scores
            W (None): Null argument for compatibility

        Returns
            Softmax (array): cross-entropy cost for each observation
        """
        P, Y = inputs
        if Y is None:
            return None

        Y = Y.astype(np.int32)

        # Normalize
        C = P.max(axis=1)
        V = P - C[:, np.newaxis]

        # log of sum of normalized exponents
        Z = np.log(np.exp(V).sum(axis=1))

        return np.mean(- V[np.arange(V.shape[0]), Y.ravel()].ravel() + Z)

    @jit(nogil=True)
    def backprop(self, inputs, H, G):
        """Backpropagate through cross-entropy loss of softmax.

        D_j [-log softmax(z)_{y=i}] = D[- z_i + log sum exp(z_j)]
                                    = {
                                        exp(z_j) / sum exp(z_k)      j neq i
                                        -1 + exp(z_i) / sum exp(z_k) else
                                       }


        Args
        X (None): Null argument for compatibility
        W (None): Null argument for compatibility
        H (array): the value of the ReLu (max{0, A})
        G (array): Incoming gradient

        Returns
        dL (list):
            [G, None], where G is the gradient of the cross-entropy of
            the softmax. This is the softmax probability for each z_j \neq i,
            and the softmax probability -1 for z_i.
        """
        P, Y = inputs
        # Normalize the Softmax
        C = P.max(axis=1)
        V = P - C[:, np.newaxis]
        V = np.exp(V)
        G = V / V.sum(axis=1)[:, np.newaxis]

        # Subtract 1 from the true label probabilities
        Y = Y.astype(np.int32).ravel()
        G[np.arange(G.shape[0]), Y] -= 1

        return [(1 / G.shape[0]) * G, None]
