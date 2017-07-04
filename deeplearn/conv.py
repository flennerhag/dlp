"""
Convolutional gates.
"""

import numpy as np
from numba import jit
from deeplearn.funcs import InGate, Processing


class Flatten(Processing):

    """Flatten 3D input.
    """

    def __init__(self):
        pass

    @jit(nogil=True)
    def forward(self, inputs):
        """Return flattened array."""
        X, = inputs
        return X.reshape(X.shape[0], -1)

    @jit(nogil=True)
    def backprop(self, inputs, state, G):
        """Reshape to 3D array."""
        X, = inputs
        return [G.reshape(*X.shape)]


class ConvStack(Processing):

    """Stack Convolutions.
    """

    def __init__(self):
        pass

    @jit(nogil=True)
    def forward(self, inputs):
        """Return 3D array."""
        return np.concatenate(inputs, 3)

    def backprop(self, inputs, H, G):
        """Split 3D array."""
        # This method does not work with @jit. However, it's merely returning
        # views of G so the computational cost is nil.
        return [G[..., i:i+1] for i in range(G.shape[3])]


class Convolve(InGate):

    """Gate for convolution."""

    def __init__(self, stride=1, pad=1):
        super(Convolve, self).__init__()
        self.stride = stride
        self.pad = pad

    @jit(nogil=True)
    def forward(self, inputs):
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
        X, W = inputs
        assert len(X.shape) == 4
        C, Z, out_shape = self._stretch(X, W)

        # Convolve and reshape
        O = np.empty(out_shape)
        for i in range(X.shape[0]):
            o = C.dot(Z[i, ...].squeeze())
            O[i, ...] = o.reshape(*out_shape[1:])

        return O

    @jit(nogil=True)
    def backprop(self, inputs, H, G):
        """Backpropagate through MatMul given incoming gradient G.

        Args
        X (None): null argument for compatibility
        W (array): Right matrix from addition
        H (None): null argument for compatibility
        G (array): Incoming gradient

        Returns
        dConv (list): gradient of the filter
        """
        X, W = inputs
        C, Z, out_shape = self._stretch(X, W)

        # Reshape the gradient to align with the matrix multiplication
        DW = 0
        DX = list()
        for i in range(X.shape[0]):
            Q = G[i, ...].reshape(1, Z.shape[2])
            # Get MatMul gradients
            DW += Q.dot(Z[i, ...].squeeze().T)
            dx = C.T.dot(Q)
            DX.append(dx)
        DX = np.vstack(DX)
        DW = DW.reshape(*W.shape)

        d = self._get_inp_shape(X)
        DX = self._collapse(d, DX, W)

        return [DX, DW]

    def _stretch(self, X, W):
        """Stretch filter and input for matrix multiplication."""
        # Ravel the filter
        C = W.reshape(1, np.prod(W.shape))

        # Prep the input
        V = X.copy()

        if self.pad:
            pad = ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0))
            V = np.pad(V, pad, "constant") if self.pad != 0 else X

        # Expand input
        samples = V.shape[0]
        conv_rows = int((V.shape[1] - W.shape[0]) / self.stride + 1)
        conv_cols = int((V.shape[2] - W.shape[1]) / self.stride + 1)

        convs = []
        for j in range(0, V.shape[1] - W.shape[0] + self.stride, self.stride):
            for i in range(0, V.shape[2] - W.shape[1] + self.stride, self.stride):

                start_row, stop_row = j, j + W.shape[0]
                start_col, stop_col = i, i + W.shape[1]

                z = V[:, start_row:stop_row, start_col:stop_col, :]
                z = z.reshape(samples, -1)
                convs.append(z)

        try:
            Z = np.stack(convs, 2)
        except ValueError:
            raise ValueError("Incompatible filter dimensions (%s, %s) for "
                             "input of size." % (X.shape[1] + self.pad,
                                                 X.shape[2] + self.pad))

        return C, Z, (samples, conv_rows, conv_cols, 1)

    def _collapse(self, dim, Dx, W):
        """Collapse expanded input to original dim.
        """
        V = np.zeros(dim, dtype=np.float32)

        c = 0
        for j in range(0, V.shape[1] - W.shape[0] + self.stride, self.stride):
            for i in range(0, V.shape[2] - W.shape[1] + self.stride, self.stride):

                z = Dx[..., c]

                start_col, stop_col = i, i + W.shape[1]
                start_row, stop_row = j, j + W.shape[0]
                r = V[:, start_row:stop_row, start_col:stop_col, :]
                V[:, start_row:stop_row, start_col:stop_col, :] = z.reshape(*r.shape)

                c += 1

        if self.pad:
            V = V[:, self.pad:-self.pad, self.pad:-self.pad, :]

        return V

    def _get_inp_shape(self, X):
        """Get shape for V(X) during backprop."""
        d = X.shape

        if self.pad:
            d = (d[0], d[1] + 2 * self.pad, d[2] + 2 * self.pad, d[3])

        return d


class Offset(InGate):

    """Offset (bias) gate.
    """

    def __init__(self):
        super(Offset, self).__init__()

    @jit(nogil=True)
    def forward(self, inputs):
        """Add bias to each of n filter.

        Args
            X (array): filter of dim L x W x D
            W (array): bias of dim D

        Returns
            H (array): X + W, where addition is on the third dimension.
        """
        X, W = inputs
        assert X.shape[3] == W.shape[1]

        C = X.copy()
        for i in range(W.shape[0]):
            C[..., i] += W[0, i]

            return C

    @jit(nogil=True)
    def backprop(self, inputs, H, G):
        """Backpropagate through the bias."""
        _, W = inputs
        grads = []
        for i in range(W.shape[1]):
            g = G[..., i]
            grads.append(g.sum())

        return [G, np.array(grads)]
