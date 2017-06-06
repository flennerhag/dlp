import numpy as np
from graph import ComputationalGraph, Node
from funcs import ReLu, MatAdd, MatMul, Norm

OBS = 10
FEATS_0 = 3
FEATS_1 = 4
FEATS_2 = 2

##########
# Data
X = np.random.random((OBS, FEATS_0))
y = np.ones(OBS)
y[:5] = 0

##########
# Hidden layers
W1 = 0.1 * np.arange(FEATS_0 * FEATS_1).reshape(FEATS_0, FEATS_1)
W1 -= 2 * W1.mean().mean()
b1 = np.ones(FEATS_1).reshape(1, FEATS_1)

W2 = 0.1 * np.arange(FEATS_1 * FEATS_2).reshape(FEATS_1, FEATS_2)
W2 -= W2.mean().mean()
b2 = np.ones(FEATS_2).reshape(1, FEATS_2)

##########
# Output layer
W3 = 0.1 * np.arange(FEATS_2 * 1).reshape(FEATS_2, 1)
W3 -= W3.mean().mean()
b3 = np.array(0)


##########
# Graph
graph = ComputationalGraph()

node1 = Node(W1, MatMul())
node2 = Node(b1, MatAdd())
node3 = Node(None, ReLu())

node4 = Node(W2, MatMul())
node5 = Node(b2, MatAdd())
node6 = Node(None, ReLu())

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
H2 = np.fmax(H1.dot(W2) + b2, 0)
H3 = np.fmax(H2.dot(W3) + b3, 0)
H4 = np.dot(H3.ravel() - y, H3.ravel() - y)
H5 = H4 / (2 * H3.shape[0])

# Check computation
np.testing.assert_array_equal(H5, graph.nodes[-1].state)

##########
# Backward prop
graph.backprop()

D4 = H3.reshape(10, 1).T
D3 = H2.T
D2 = H1.T
D1 = X.T


for node in graph.nodes:
    try:
        print(node.param.shape, node.grad_param.shape, node.grad_input.shape)
    except:
        pass
