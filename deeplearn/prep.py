"""
Simple preprocessing.
"""

import numpy as np


class Preprocessing(object):
    """Preprocessing meta class."""


class Decorrelate(Preprocessing):
    """Decorrelation transformer.
    """

    def __init__(self, n_components=None, whitening_smoother =1e-5):
        self.n_components = n_components
        self.whitening_smoother = whitening_smoother
        self.U = None
        self.S = None

    def fit(self, X):
        """Fit SVD to X."""
        n = self.n_components if self.n_components is not None else X.shape[1]
        X -= X.mean(axis=0)
        C = X.T.dot(X) / X.shape[0]
        U, S, _ = np.linalg.svd(C)

        self.U = U[:, :n]; self.S = S[:, :n]

    def PCA(self, X):
        """Change of coordiantes into Eigenbasis."""
        return X.dot(self.U)

    def Whiten(self, X):
        """PCA of X."""
        X = self.PCA(X)
        return X / np.sqrt(self.S + self.whitening_smoother)
