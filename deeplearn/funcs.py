"""
Functions for forward and backward calculation of gradients.
"""

import numpy as np
from numba import jit

###############################################################################
# Auxiliary functions


@jit(nogil=True)
def sigmoid(x):
    """The sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


@jit(nogil=True)
def probmax(x, zero_index=True):
    """Function selecting the class label with highest probability."""
    i = 0 if zero_index else 1
    return x.argmax(axis=1) + i

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
    def forward(self, X):
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
    def backprop(self, X, H, G):
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
        return [np.multiply(H, (1 - H))]


class ReLu(Activation):

    """Gate for RelU activation.
    """

    def __init__(self):
        pass

    @jit(nogil=True)
    def forward(self, X):
        """Calculate the ReLu in a forward pass.

        Args
        inputs (tuple):
            X (array): Output to activate
            W (None): Null argument for compatibility

        Returns
        Relu (array): output array of size [n_samples, n_out_features]
        """
        return np.maximum(X, 0)

    @jit(nogil=True)
    def backprop(self, X, H, G):
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
        V = 1 * (H >= 0)
        O = np.multiply(V, G)
        return [O]


class PReLu(Activation):

    """Gate for RelU activation.
    """

    def __init__(self):
        pass

    @jit(nogil=True)
    def forward(self, X, alpha):
        """Calculate the ReLu in a forward pass.

        Args
        inputs (tuple):
            X (array): Output to activate
            W (None): Null argument for compatibility

        Returns
        Relu (array): output array of size [n_samples, n_out_features]
        """
        return np.where(X >= 0, X, alpha*X)

    @jit(nogil=True)
    def backprop(self, X, alpha, H, G):
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
        Z = H < 0
        V = np.where(H >= 0, 1, alpha)
        return [np.multiply(V, G), np.multiply(V, Z).sum()]


class SeLu(Activation):

    """Gate for SeLu activation.
    """

    def __init__(self,
                 l=1.0507009873554804934193349852946,
                 a=1.6732632423543772848170429916717):
        self.a = a
        self.l = l

    @jit(nogil=True)
    def forward(self, X):
        """Calculate the ReLu in a forward pass.

        Args
        inputs (tuple):
            X (array): Output to activate
            W (None): Null argument for compatibility

        Returns
        SeLu (array): output array of size [n_samples, n_out_features]
        """
        return self.l * np.where(X > 0.0, X, self.a * (np.exp(X) - 1))

    @jit(nogil=True)
    def backprop(self, X, H, G):
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
        V = np.where(X > 0.0, self.l, self.l * self.a * np.exp(X))
        return [np.multiply(V, G)]




###############################################################################
# Processing

class Processing(object):

    """Meta class for inter-stage processing units.
    """


class DropOut(Processing):

    """Dropout gate.

    This gate implements inverted Dropout.

    Note:
        If this gate is connected to an activation unit, the result is DropOut.
        If connected to a weight matrix, the result is DropConnect.
    """

    def __init__(self, p=0.5, dist=None, scale=False):
        self.train = True

        self.W = 0
        self.p = p

        self.scale = scale
        if dist is not None:
            self.dist = dist
        else:
            self.dist = np.random.rand

    @jit(nogil=True)
    def forward(self, X):
        """Randomly switch samples off."""
        if not self.train:
            return X  # we scale by 1/p during training instead
        else:
            H = self.dist(*X.shape)

            if self.scale:
                m = H.max()
                n = H.min()
                H -= n
                H /= m - n

            self.W = (H < self.p) / self.p
            return np.multiply(X, self.W)

    @jit(nogil=True)
    def backprop(self, X, H, G):
        """Backpropagate through Dropout."""
        # We just take the gradient and switch off the inactive units.
        return [np.multiply(self.W, G), None]


class Normalize(Processing):

    """Batch normalization gate.
    """

    def __init__(self, e=0.0001):
        self.e = e
        self.u = []
        self.s = []
        self.train = True

    @jit(nogil=True)
    def forward(self, X):
        """Normalize batch before activation."""
        if self.train:
            u = X.mean(axis=0)
            s = X.std(axis=0)
            self.u.append(u); self.s.append(s)
        else:
            u = np.mean(self.u, axis=0)
            s = np.mean(self.s, axis=0)

        Y = X - u
        Y /= (s + self.e)
        return Y

    @jit(nogil=True)
    def backprop(self, X, H, G):
        """Backprop through normalization wrt X."""
        u = self.u[-1]
        s = self.s[-1]
        e = self.e
        M = X - u
        N = X.shape[0]

        s_inv = (s + e) ** (-1/2)

        Ds = np.multiply(G, M).sum(axis=0)
        Ds = np.multiply(Ds, -(1/2) * s_inv ** 3)

        Du = np.multiply(G.sum(axis=0), -1 * s_inv)
        Du -= (2 / N) * np.multiply(M.sum(axis=0), Ds)

        Dx = np.multiply(G, s_inv) + (2 / N) * np.multiply(M, Ds) + Du / N

        return [Dx]


###############################################################################
# Ingoing units


class InGate(object):

    """Meta class for ingoing hidden units.
    """
    def __init__(self):
        pass


class VecMul(InGate):

    """Matrix to vector broadcasted multiplication.

    Used with batch normalization.
    """
    def __init__(self):
        super(VecMul, self).__init__()

    @jit(nogil=True)
    def forward(self, X, W):
        """Forward multiplication with respect to a parameter matrix W.

        Args
            X (array): Left matrix for multiplication
            W (array): Right vector for multiplication

        Returns
            C (array): broadcaster multiplication X * W
        """
        return np.multiply(X, W)

    @jit(nogil=True)
    def backprop(self, X, W, H, G):
        """Backpropagate through MatMul given incoming gradient G.

        Args
        X (array): Left matrix from multiplication
        W (array): Right matrix from multiplication
        H (None): null argument for compatibility
        G (array): Incoming gradient

        Returns
        dVMul (list): [A, B], where A is the derivative wrt X, and B is
            the derivative wrt W.
        """
        return [np.multiply(G, W), np.multiply(G, X).sum(axis=0)]


class MatMul(InGate):

    """Gate for matrix multiplication.
    """

    def __init__(self):
        super(MatMul, self).__init__()

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
        super(MatAdd, self).__init__()

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
