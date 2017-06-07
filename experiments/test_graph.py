import numpy as np

from sklearn.datasets import make_classification

from deeplearn.train import Trainer
from deeplearn.opts import GradientDescent
from deeplearn.graph import ComputationalGraph, Node

from deeplearn.cost_func import Norm
from deeplearn.funcs import MatAdd, MatMul, PReLu, ReLu
from deeplearn.init import init_bias, init_weights

np.random.seed(10)

OBS = 100
FEATS_0 = 5
FEATS_1 = 4
FEATS_2 = 2

##########
# Data
X, y = make_classification(OBS, FEATS_0)

##########
# Hidden layers
W1 = init_weights(FEATS_0, FEATS_1)
W2 = init_weights(FEATS_1, FEATS_2)

b1 = init_bias(FEATS_1, 0.)
b2 = init_bias(FEATS_2, 0.)

##########
# Output layer
W3 = init_weights(FEATS_2, 1)
b3 = init_bias(1, 0.)

##########
# Graph
graph = ComputationalGraph()

node1 = Node(W1, MatMul())
node2 = Node(b1, MatAdd())
node3 = Node(None, ReLu())

node4 = Node(W2, MatMul())
node5 = Node(b2, MatAdd())
node6 = Node(None, PReLu())

node7 = Node(W3, MatMul())
node8 = Node(b3, MatAdd())
node9 = Node(None, ReLu())
node10 = Node(y, Norm())

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

##########
# Forward prop
graph.forward(X)

H1 = np.fmax(X.dot(W1) + b1, 0)
H2 = np.fmax(H1.dot(W2) + b2, 0.01*(H1.dot(W2) + b2))
H3 = np.fmax(H2.dot(W3) + b3, 0)
H4 = np.dot(H3.ravel() - y, H3.ravel() - y)
H5 = H4 / (2 * H3.shape[0])

# Check forward computation
np.testing.assert_array_almost_equal(H1, graph.nodes[3].state)
np.testing.assert_array_almost_equal(H2, graph.nodes[6].state)
np.testing.assert_array_almost_equal(H3, graph.nodes[9].state)
np.testing.assert_array_almost_equal(H5, graph.nodes[10].state)


# Gradient check
graph.backprop()

graph.clear()

