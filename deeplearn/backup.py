import numpy as np

from sgd import SGD
from graph import ComputationalGraph, Node
from funcs import ReLu, MatAdd, MatMul, Norm

from sklearn.datasets import make_classification

import matplotlib.pyplot as plt


np.random.seed(2020)

OBS = 1000000
ITERS = 10000
FEATS_0 = 10
FEATS_1 = 50
FEATS_2 = 10
FEATS_3 = 20
FEATS_4 = 4

##########
# Data
X, y = make_classification(OBS, FEATS_0, 5)

##########
# Hidden layers
W1 = np.random.random((FEATS_0, FEATS_1))
b1 = np.ones(FEATS_1, dtype=np.float32).reshape(1, FEATS_1)

W2 = np.random.random((FEATS_1, FEATS_2))
b2 = np.ones(FEATS_2, dtype=np.float32).reshape(1, FEATS_2)

W3 = np.random.random((FEATS_2, FEATS_3))
b3 = np.ones(FEATS_3, dtype=np.float32).reshape(1, FEATS_3)

W4 = np.random.random((FEATS_3, FEATS_4))
b4 = np.ones(FEATS_4, dtype=np.float32).reshape(1, FEATS_4)

##########
# Output layer
W5 = np.random.random((FEATS_4, 1))
b5 = np.array(0, dtype=np.float32)

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

node10 = Node(W4, MatMul())
node11 = Node(b4, MatAdd())
node12 = Node(None, ReLu())

node13 = Node(W5, MatMul())
node14 = Node(b5, MatAdd())
node15 = Node(None, ReLu())
node16 = Node(y, Norm())

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

###############################################################################
# SGD

sgd = SGD(graph, 1e-9, batch_size=20, eval_size=10000, decay=1e-4)

sgd.run(X, y, ITERS)

norm, train_loss, test_loss = sgd.norm, sgd.train_loss, sgd.test_loss


###############################################################################
# VISUAL

BURNIN = int(np.floor(ITERS*0.05))

f, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(range(BURNIN, ITERS), train_loss[BURNIN:], linewidth=0.5, color='k',
           label="train error")
ax[0].plot(range(BURNIN, ITERS, 10), test_loss[int(BURNIN/10):], linewidth=0.5,
           color='r', label="test error")
ax[0].set_title("Loss")
ax[0].legend(frameon=False)
ax[1].plot(range(BURNIN, ITERS), norm[BURNIN:], linewidth=0.5, color='k')
ax[1].set_title("Gradient norm")
