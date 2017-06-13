"""
MNIST dataset.
"""

import os
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.datasets import get_data_home
from sklearn.externals.joblib import Memory
from sklearn.utils import check_array

from deeplearn.funcs import probmax
from deeplearn.train import Trainer
from deeplearn.opts import RMSProp
from deeplearn.networks import Sequential
from deeplearn.viz import plot_train_scores


# Memoize the data extraction and memory map the resulting
# train / test splits in readonly mode
memory = Memory(os.path.join(get_data_home(), 'mnist_benchmark_data'),
                mmap_mode='r')


@memory.cache
def load_data(dtype=np.float32, order='C'):
    """Load the data, then cache and memmap the train/test split"""
    ######################################################################
    # Load dataset
    print("Loading dataset...")
    data = fetch_mldata('MNIST original')
    X = check_array(data['data'], dtype=dtype, order=order)
    y = data["target"]

    # Normalize features
    X = X / 255

    # Create train-test split (as [Joachims, 2006])
    print("Creating train-test split...")
    n_train = 60000
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test


###############################################################################
ITERS = 500
STEPSIZE = 10


def build_net(drop):
    """Build network with or without dropout."""
    net = Sequential()
    net.add_fc(784, 1500)
    net.add_fc(1500, 1000, dropout=drop)
    net.add_fc(1000, 500, dropout=drop)
    net.add_fc(500, 10)
    net.add_cost("softmax")

    return net


def err(y, p):
    """Error of network"""
    y = y.ravel()
    p = probmax(p)
    return (p != y).sum() / y.shape[0]


def shuffle(X, y):
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]
    return X, y


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_data(order="C")

    print("")
    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("number of features:".ljust(25), X_train.shape[1]))
    print("%s %d" % ("number of classes:".ljust(25), np.unique(y_train).size))
    print("%s %s" % ("data type:".ljust(25), X_train.dtype))
    print("%s %d (size=%dMB)" % ("number of train samples:".ljust(25),
                                 X_train.shape[0], int(X_train.nbytes / 1e6)))
    print("%s %d (size=%dMB)" % ("number of test samples:".ljust(25),
                                 X_test.shape[0], int(X_test.nbytes / 1e6)))

    print()
    print("Training Network")
    print("====================")

    net = build_net(True)
    opt = RMSProp(net, decay=1e-6)
    trainer = Trainer(net, opt,
                      batch_size=100,
                      eval_size=1000,
                      eval_ival=STEPSIZE,
                      eval_metric=err,
                      )

    X_train, y_train = shuffle(X_train, y_train)

    trainer.train(X_train, y_train, ITERS)
    plot_train_scores(trainer, 0)

