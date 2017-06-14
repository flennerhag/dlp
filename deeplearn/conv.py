"""
Convolutional gates.
"""

import numpy as np
from numba import jit
from .funcs import InGate, Processing


class ConvStack(Processing):

    """Stack Convolutions.
    """

    def __init__(self):
        pass

    @jit(nogil=True)
    def forward(self, *args):
        """Return 3D array."""
        return np.dstack(args)

    def backprop(self, X, W, H, G):
        """Split 3D array."""
        # This method does not work with @jit. However, it's merely returning
        # views of G so the computational cost is nil.
        return [G[:, :, i] for i in range(G.shape[2])]


class Convolve(InGate):

    """Gate for convolution."""

    def __init__(self, stride=1, pad=1):
        self.stride = stride
        self.pad = pad

#    @jit(nogil=True)
    def forward(self, X, W):
        """Forward addition with respect to a parameter matrix W.

        Args
            X (array): Input array of dim (V_1, H_1, D_1)
            W (array): Filter of dim (R_1, C_1, N)

        Returns
            C (array): convolution of dim (V_2, H_2, N), where
             V_2 = (V_1 - R_1 + 2P)/S + 1
             H_2 = (H_1 - C_1 + 2P)/S + 1

             and P = amount of zero padding while S = stride length.
        """
        C, Z, out_shape = self._stretch(X, W)

        # Convolve and reshape
        O = C.dot(Z)
        return O.reshape(*out_shape)

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
        C, Z, out_shape = self._stretch(X, W)

        # Reshape the gradient to align with the matrix multiplication
        Q = G.reshape(C.shape[0], Z.shape[1])

        # Get MatMul gradients
        DW = Q.dot(Z.T)
        DX = C.T.dot(Q)

        DW = DW.reshape(*W.shape)

        d = self._get_inp_shape(X)
        DX = self._collapse(d, DX, W)

        return [DX, DW]

    @jit(nogil=True)
    def _stretch(self, X, W):
        """Stretch filter and input for matrix multiplication."""
        # Ravel the filter
        C = W.reshape(1, np.prod(W.shape))

        # Prep the input
        V = X.copy()
        if len(V.shape) == 1:
            V = V[..., np.newaxis, np.newaxis]
        elif len(V.shape) == 2:
            V = V[..., np.newaxis]

        if self.pad:
            pad = ((self.pad, self.pad), (self.pad, self.pad), (0, 0))
            V = np.pad(V, pad, "constant") if self.pad != 0 else X

        # Expand input
        conv_rows = int((V.shape[0] - W.shape[0]) / self.stride + 1)
        conv_cols = int((V.shape[1] - W.shape[1]) / self.stride + 1)

        c = 0
        convs = []
        for j in range(0, V.shape[0] - W.shape[0] + self.stride, self.stride):
            for i in range(0, V.shape[1] - W.shape[1] + self.stride,
                           self.stride):

                start_row, stop_row = j, j + W.shape[0]
                start_col, stop_col = i, i + W.shape[1]
                z = V[start_row:stop_row, start_col:stop_col, :].ravel()
                convs.append(z)
                c += 1

        return C, np.column_stack(convs), (conv_rows, conv_cols)

    @jit(nogil=True)
    def _collapse(self, dim, Dx, W):
        """Collapse expanded input to original dim.
        """
        V = np.empty(dim)

        c = 0
        for j in range(0, V.shape[0] - W.shape[0] + self.stride, self.stride):
            for i in range(0, V.shape[1] - W.shape[1] + self.stride,
                           self.stride):

                z = Dx[:, c]

                start_col, stop_col = i, i + W.shape[1]
                start_row, stop_row = j, j + W.shape[0]
                r = V[start_row:stop_row, start_col:stop_col, :]
                r += z.reshape(*r.shape)

                c += 1

        if self.pad:
            V = V[self.pad:-self.pad, self.pad:-self.pad, :]

        return V

    @jit(nogil=True)
    def _get_inp_shape(self, X):
        """Get shape for V(X) during backprop."""
        d = X.shape
        if len(d) == 1:
            d = (d, 1, 1)
        elif len(d) == 2:
            d = (d[0], d[1], 1)

        if self.pad:
            d = (d[0] + 2 * self.pad, d[1] + 2 * self.pad, d[2])

        return d


class Offset(InGate):

    """Offset (bias) gate.
    """

    def __init__(self):
        pass

    @jit(nogil=True)
    def forward(self, X, W):
        """Add bias to each of n filter.

        Args
            X (array): filter of dim L x W x D
            W (array): bias of dim D

        Returns
            H (array): X + W, where addition is on the third dimension.
        """
        if len(X.shape) == 2:
            return X + W
        else:
            assert X.shape[2] == W.shape[0]

            C = X.copy()
            for i in range(W.shape[0]):
                C[:, :, i] += W[i]

            return C

    @jit(nogil=True)
    def backprop(self, X, W, H, G):
        """Backpropagate through the bias."""
        if len(X.shape) == 2:
            # Standard MatAdd
            dW = np.ones((1, G.shape[0]))
            return [G, dW.dot(G)]

        else:
            # Get gradient for each filter
            grads = []
            for i in range(W.shape[0]):
                g = G[:, :, i]
                dW = np.ones((1, g.shape[0]))
                grads.append(dW.dot(g))

            return [G, np.array(grads)]
