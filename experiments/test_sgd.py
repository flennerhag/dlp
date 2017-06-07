import numpy as np

from deeplearn.train import Trainer
from deeplearn.opts import GradientDescent, Momentum, Nesterov, RMSProp, Adam
from deeplearn.graph import ComputationalGraph, Node

from deeplearn.cost_func import Norm
from deeplearn.funcs import MatAdd, MatMul, PReLu, ReLu, sigmoid
from deeplearn.init import init_bias, init_weights
from deeplearn.viz import plot_train_scores

from sklearn.datasets import make_classification

###############################################################################
# Data and Network size

OBS = int(1e6)
ITERS = 1000
STEPSIZE = 10


FEATS_0 = 10
FEATS_1 = 1000
FEATS_2 = 2000
FEATS_3 = 6
FEATS_4 = 500
FEATS_5 = 12

###############################################################################
# Data
X, y = make_classification(OBS, FEATS_0, 5)

X -= X.mean(axis=0)
X /= X.std(axis=0)

###############################################################################
# Network

# Weights
W1 = init_weights(FEATS_0, FEATS_1)
W2 = init_weights(FEATS_1, FEATS_2)
W3 = init_weights(FEATS_2, FEATS_3)
W4 = init_weights(FEATS_3, FEATS_4)
W5 = init_weights(FEATS_4, FEATS_5)

# Biases
b1 = init_bias(FEATS_1, 0.01)
b2 = init_bias(FEATS_2, 0.01)
b3 = init_bias(FEATS_3, 0.01)
b4 = init_bias(FEATS_4, 0.01)
b5 = init_bias(FEATS_5, 0.01)

# Activation non-linearity
a1 = 0.01
a2 = 0.01
a3 = 0.01
a4 = 0.01
a5 = 0.01
a6 = 0.01

# Output layer
W6 = init_weights(FEATS_5, 1)
b6 = np.array([[0]], dtype=np.float32)

# Graph
graph = ComputationalGraph()

node1 = Node(W1, MatMul())
node2 = Node(b1, MatAdd())
node3 = Node(a1, PReLu(a1))

node4 = Node(W2, MatMul())
node5 = Node(b2, MatAdd())
node6 = Node(a2, PReLu(a2))

node7 = Node(W3, MatMul())
node8 = Node(b3, MatAdd())
node9 = Node(a3, PReLu(a3))

node10 = Node(W4, MatMul())
node11 = Node(b4, MatAdd())
node12 = Node(a4, PReLu(a4))

node13 = Node(W5, MatMul())
node14 = Node(b5, MatAdd())
node15 = Node(a5, PReLu(a5))

node16 = Node(W6, MatMul())
node17 = Node(b6, MatAdd())
node18 = Node(a6, PReLu(a6))

node19 = Node(y, Norm())

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

###############################################################################
# Stochastic Gradient Descent


def acc(y, p, sig=False, C=0.5):
    """Accuracy of network"""
    y = y.ravel()
    p = p.ravel()
    if sig:
        p = sigmoid(p)

    p = 1 * (p > C)
    return (p == y).sum() / y.shape[0]

opt1 = GradientDescent(graph, 5*1e-3, 1e-8)
opt2 = Momentum(graph, 5*1e-3, 0.9, 1e-8)
opt3 = Nesterov(graph, 5*1e-3, 0.9, 1e-8)
opt4 = RMSProp(graph, 5*1e-3, 0.9, 1e-8)
opt5 = Adam(graph, 5*1e-3)

sgd = Trainer(graph,
              opt5,
              batch_size=100,
              eval_size=1000,
              eval_ival=STEPSIZE,
              eval_metric=acc,
              )

sgd.train(X, y, ITERS)

plot_train_scores(sgd)
