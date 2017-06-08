"""
MNIST dataset.
"""

import os
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.datasets import get_data_home
from sklearn.externals.joblib import Memory
from sklearn.utils import check_array

from deeplearn.train import Trainer
from deeplearn.opts import RMSProp
from deeplearn.graph import ComputationalGraph, Node

from deeplearn.cost_func import Softmax
from deeplearn.funcs import MatAdd, MatMul, PReLu, probmax, DropOut
from deeplearn.init import init_bias, init_weights
from deeplearn.viz import plot_train_scores


# Memoize the data extraction and memory map the resulting
# train / test splits in readonly mode
memory = Memory(os.path.join(get_data_home(), 'mnist_benchmark_data'),
                mmap_mode='r')


@memory.cache
def load_data(dtype=np.float32, order='F'):
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
ITERS = 2000
STEPSIZE = 10

FEATS_0 = 784
FEATS_1 = 1200
FEATS_2 = 1000
FEATS_3 = 800
FEATS_4 = 500
FEATS_5 = 100

# Network
def build_net():
    # Weights
    W1 = init_weights(FEATS_0, FEATS_1)
    W2 = init_weights(FEATS_1, FEATS_2)
    W3 = init_weights(FEATS_2, FEATS_3)
    W4 = init_weights(FEATS_3, FEATS_4)
    W5 = init_weights(FEATS_4, FEATS_5)

    # Biases
    b1 = init_bias(FEATS_1)
    b2 = init_bias(FEATS_2)
    b3 = init_bias(FEATS_3)
    b4 = init_bias(FEATS_4)
    b5 = init_bias(FEATS_5)

    # Output layer
    W6 = init_weights(FEATS_5, 10)
    b6 = init_bias(10)

    # Graph
    graph = ComputationalGraph()

    node1 = Node(W1, MatMul())
    node2 = Node(b1, MatAdd())
    node3 = Node(None, PReLu())
    node4 = Node(None, DropOut())

    node5 = Node(W2, MatMul())
    node6 = Node(b2, MatAdd())
    node7 = Node(None, PReLu())
    node8 = Node(None, DropOut())

    node9 = Node(W3, MatMul())
    node10 = Node(b3, MatAdd())
    node11 = Node(None, PReLu())
    node12 = Node(None, DropOut())

    node13 = Node(W4, MatMul())
    node14 = Node(b4, MatAdd())
    node15 = Node(None, PReLu())
    node16 = Node(None, DropOut())

    node17 = Node(W5, MatMul())
    node18 = Node(b5, MatAdd())
    node19 = Node(None, PReLu())
    node20 = Node(None, DropOut())

    node21 = Node(W6, MatMul())
    node22 = Node(b6, MatAdd())
    node23 = Node(None, PReLu())

    node24 = Node(None, Softmax())

    graph.add_node(node1, 0)
    graph.add_node(node2, 1)
    graph.add_node(node3, 2)
    graph.add_node(node4, 3)

    graph.add_node(node5, 4)
    graph.add_node(node6, 5)
    graph.add_node(node7, 6)
    graph.add_node(node8, 7)

    graph.add_node(node9, 8)
    graph.add_node(node10, 9)
    graph.add_node(node11, 10)
    graph.add_node(node12, 11)

    graph.add_node(node13, 12)
    graph.add_node(node14, 13)
    graph.add_node(node15, 14)
    graph.add_node(node16, 15)

    graph.add_node(node17, 16)
    graph.add_node(node18, 17)
    graph.add_node(node19, 18)
    graph.add_node(node20, 19)

    graph.add_node(node21, 20)
    graph.add_node(node22, 21)
    graph.add_node(node23, 22)

    graph.add_node(node24, 23)

    return graph


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

    graph = build_net()
    opt = RMSProp(graph, lr=1e-3, decay=1e-6)
    sgd = Trainer(graph,
                  opt,
                  batch_size=1000,
                  eval_size=10000,
                  eval_ival=STEPSIZE,
                  eval_metric=err,
                  )
    X_train, y_train = shuffle(X_train, y_train)

    sgd.train(X_train, y_train, ITERS)
    plot_train_scores(sgd, 0)

