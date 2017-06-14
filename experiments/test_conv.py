
import numpy as np
from deeplearn.funcs import ReLu
from deeplearn.cost_func import Norm
from deeplearn.networks import Sequential
from deeplearn.init import init_filter, init_bias
from deeplearn.conv import Convolve, ConvStack, Offset
from deeplearn.graph import ComputationalGraph, Gate, Variable, Input, Output

np.random.seed(10)
X = np.arange(50).reshape(5, 5, 2)
y = np.ones(1)

###############################################################################
# Network
np.random.seed(10)

net = Sequential()
net.add_conv((3, 2, 2), 2)
net.add_cost("norm")

###############################################################################
# Graph
np.random.seed(10)

W1 = init_filter(3, 2, 2)
W2 = init_filter(3, 2, 2)

b1 = init_bias(1, 0.)[0, 0]
b2 = init_bias(1, 0.)[0, 0]

graph = ComputationalGraph()
node_r0 = Input()

node_w1 = Variable(W1)
node_b1 = Variable(b1)
node_w2 = Variable(W2)
node_b2 = Variable(b2)

node_f1 = Gate(Convolve())
node_a1 = Gate(Offset())

node_f2 = Gate(Convolve())
node_a2 = Gate(Offset())

node_c = Gate(ConvStack())
node_r = Gate(ReLu())

node_y = Output(y)
node_L = Gate(Norm())


graph.add_node(node_r0)                      # Node 0
graph.add_node(node_w1)                      # Node 1
graph.add_node(node_b1)                      # Node 2
graph.add_node(node_w2)                      # Node 1
graph.add_node(node_b2)                      # Node 2

graph.add_node(node_f1, [node_r0, node_w1])
graph.add_node(node_a1, [node_f1, node_b1])
graph.add_node(node_f2, [node_r0, node_w2])
graph.add_node(node_a2, [node_f2, node_b2])

graph.add_node(node_c, [node_a1, node_a2])
graph.add_node(node_r, [node_c])

graph.add_node(node_y)
graph.add_node(node_L, [node_r, node_y])

graph.forward(X, y)
graph.backprop()

net.graph.forward(X, y)
net.graph.backprop()

for n, m in zip(graph.get_nodes(), net.graph.get_nodes()):
    np.testing.assert_array_equal(n.state, m.state)
