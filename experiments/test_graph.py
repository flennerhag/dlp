import numpy as np

from sklearn.datasets import make_classification

from deeplearn.graph import ComputationalGraph, Gate, Variable, Input, Output

from deeplearn.cost_func import Norm
from deeplearn.funcs import MatAdd, MatMul, ReLu
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

node_r0 = Input()

node_w1 = Variable(W1)
node_b1 = Variable(b1)
node_m1 = Gate(MatMul())
node_a1 = Gate(MatAdd())
node_r1 = Gate(ReLu())

node_w2 = Variable(W2)
node_b2 = Variable(b2)
node_m2 = Gate(MatMul())
node_a2 = Gate(MatAdd())
node_r2 = Gate(ReLu())

node_w3 = Variable(W3)
node_b3 = Variable(b3)
node_m3 = Gate(MatMul())
node_a3 = Gate(MatAdd())
node_r3 = Gate(ReLu())

node_y = Output(y)
node_L = Gate(Norm())

# Initialize data node
graph.add_node(node_r0)                      # Node 0
graph.add_node(node_w1)                      # Node 1
graph.add_node(node_b1)                      # Node 2
graph.add_node(node_m1, [node_r0, node_w1])  # Node 3
graph.add_node(node_a1, [node_m1, node_b1])  # Node 4
graph.add_node(node_r1, [node_a1])           # Node 5

graph.add_node(node_w2)                      # Node 6
graph.add_node(node_b2)                      # Node 7
graph.add_node(node_m2, [node_r1, node_w2])  # Node 8
graph.add_node(node_a2, [node_m2, node_b2])  # Node 9
graph.add_node(node_r2, [node_a2])           # Node 10

graph.add_node(node_w3)                      # Node 11
graph.add_node(node_b3)                      # Node 12
graph.add_node(node_m3, [node_r2, node_w3])  # Node 13
graph.add_node(node_a3, [node_m3, node_b3])  # Node 14
graph.add_node(node_r3, [node_a3])           # Node 15

graph.add_node(node_y)                       # Node 16
graph.add_node(node_L, [node_r3, node_y])    # Node 17


##########
# Forward prop
graph.forward(X)

H1 = np.fmax(X.dot(W1) + b1, 0)
H2 = np.fmax(H1.dot(W2) + b2, 0)
H3 = np.fmax(H2.dot(W3) + b3, 0)
H4 = np.dot(H3.ravel() - y, H3.ravel() - y)
H5 = H4 / (2 * H3.shape[0])

# Check forward computation
np.testing.assert_array_almost_equal(H1, node_r1.state)
np.testing.assert_array_almost_equal(H2, node_r2.state)
np.testing.assert_array_almost_equal(H3, node_r3.state)
np.testing.assert_array_almost_equal(H5, node_L.state)


# Gradient check
graph.backprop()

graph.clear()
