"""
Functions for forward and backward calculation of gradients.
"""

import numpy as np
from numba import jit

###############################################################################
# Auxiliary functions


def sigmoid(x):
    """The sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


###############################################################################
# Activations


class Activation(object):

    """Activation meta class.
    """


class Sigmoid(Activation):

    """Gate for Sigmoid activation.
    """

    def __init__(self):
        pass

    @jit(nogil=True)
    def forward(self, X, W):
        """Calculate the sigmoid element-wise in a forward pass.

        Args
        inputs (tuple):
            X (array): Output to activate
            W (None): Null argument for compatibility

        Returns
        Sigmoid (array): output array of size [n_samples, n_out_features]
        """
        return sigmoid(X)

    @jit(nogil=True)
    def backprop(self, X, W, H, G):
        """Backpropagate through ReLu given incoming gradient G.

        Args
        X (None): Null argument for compatibility
        W (None): Null argument for compatibility
        H (array): the value of the ReLu (max{0, A})
        G (array): Incoming gradient

        Returns
        dReLu (list):
            [B, None], where B is the backpropagated gradient of ReLu. This is
            the same as G but with b_{ij} = 0 if h_{ij} < 0.
        """
        return [np.multiply(H, (1 - H)), None]


class ReLu(Activation):

    """Gate for RelU activation.
    """

    def __init__(self):
        pass

    @jit(nogil=True)
    def forward(self, X, W):
        """Calculate the ReLu in a forward pass.

        Args
        inputs (tuple):
            X (array): Output to activate
            W (None): Null argument for compatibility

        Returns
        Relu (array): output array of size [n_samples, n_out_features]
        """
        null = np.zeros(X.shape)
        return np.fmax(X, null)

    @jit(nogil=True)
    def backprop(self, X, W, H, G):
        """Backpropagate through ReLu given incoming gradient G.

        Args
        X (None): Null argument for compatibility
        W (None): Null argument for compatibility
        H (array): the value of the ReLu (max{0, A})
        G (array): Incoming gradient

        Returns
        dReLu (list):
            [B, None], where B is the backpropagated gradient of ReLu. This is
            the same as G but with b_{ij} = 0 if h_{ij} < 0.
        """
        H = 1 * (H > 0)
        return [np.multiply(H, G), None]


class PReLu(Activation):

    """Gate for RelU activation.
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    @jit(nogil=True)
    def forward(self, X, W):
        """Calculate the ReLu in a forward pass.

        Args
        inputs (tuple):
            X (array): Output to activate
            W (None): Null argument for compatibility

        Returns
        Relu (array): output array of size [n_samples, n_out_features]
        """
        return np.fmax(X, self.alpha * X)

    @jit(nogil=True)
    def backprop(self, X, W, H, G):
        """Backpropagate through ReLu given incoming gradient G.

        Args
        X (None): Null argument for compatibility
        W (None): Null argument for compatibility
        H (array): the value of the ReLu (max{0, A})
        G (array): Incoming gradient

        Returns
        dReLu (list):
            [B, None], where B is the backpropagated gradient of ReLu. This is
            the same as G but with b_{ij} = 0 if h_{ij} < 0.
        """
        Hp = 1 * (H >= 0)
        Z = H < 0
        Hm = self.alpha * Z
        H = Hp + Hm
        return [np.multiply(H, G), np.multiply(H, Z).mean()]


###############################################################################
# Connections

class Connect(object):

    """Meta class for connecting units.
    """


class DropConnect(Connect):

    """Dropout element wise.

    Class for element-wise Dropout.
    """

    def __init__(self):
        pass

    @jit(nogil=True)
    def forward(self, X, W):
        """Randomly switch samples off."""
        np.random.bernoulli()

    @jit(nogil=True)
    def backprop(self, X, W, H, G):
        return


###############################################################################
# Ingoing units


class InGate(object):

    """Meta class for ingoing hidden units.
    """


class MatMul(InGate):

    """Gate for matrix multiplication.
    """

    def __init__(self):
        pass

    @jit(nogil=True)
    def forward(self, X, W):
        """Forward multiplication with respect to a parameter matrix W.

        Args
            X (array): Left matrix for multiplication
            W (array): Right matrix for multiplication

        Returns
            C (array): product matrix XW
        """
        return X.dot(W)

    @jit(nogil=True)
    def backprop(self, X, W, H, G):
        """Backpropagate through MatMul given incoming gradient G.

        Args
        X (array): Left matrix from multiplication
        W (array): Right matrix from multiplication
        H (None): null argument for compatibility
        G (array): Incoming gradient

        Returns
        dMatMul (list): [A, B], where A is the derivative wrt X, and B is
            the derivative wrt W.
        """
        return [G.dot(W.T), X.T.dot(G)]


class MatAdd(InGate):

    """Gate for Matrix addition.
    """

    def __init__(self):
        pass

    @jit(nogil=True)
    def forward(self, X, W):
        """Forward addition with respect to a parameter matrix W.

        Args
            X (array): Left matrix for addition
            W (array): Right matrix for addition

        Returns
            C (array): sum matrix X + W
        """
        return X + W

    @jit(nogil=True)
    def backprop(self, X, W, H, G):
        """Backpropagate through MatMul given incoming gradient G.

        Args
        X (None): null argument for compatibility
        W (array): Right matrix from addition
        H (None): null argument for compatibility
        G (array): Incoming gradient

        Returns
        dMatAdd (list): [A, B], where A is G, and B is the derivative wrt W.
        """
        dW = np.ones((1, G.shape[0]))
        return [G, dW.dot(G)]
