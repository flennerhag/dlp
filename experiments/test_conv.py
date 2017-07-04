
import numpy as np
np.random.seed(1)

from deeplearn.conv import *
from deeplearn.networks import Sequential
from deeplearn.funcs import MatMul
from deeplearn.cost_func import Softmax
from deeplearn.init import init_filter, init_bias, init_weights
from deeplearn.graph import Variable, Gate, ComputationalGraph, Input, Output


X = np.random.rand(2, 4, 4, 3)
y = np.arange(2)

np.random.seed(1)
graph = ComputationalGraph()

input = Input(X)
target = Output(y)

graph.add_node(input)

filters = list()
pars = list()
weights = list()
for i in range(2):

    # Add filter
    weights.append(init_filter(4, 4, 3))
    pars.append(Variable(weights[-1]))
    graph.add_node(pars[-1])

    # Add convolution
    conv = Gate(Convolve(1, 1))
    graph.add_node(conv, parent_nodes=[input, pars[-1]])

    # Record convolution for stacking
    filters.append(graph.nodes[-1])

# Stack filter outputs
stack = Gate(ConvStack())
graph.add_node(stack, parent_nodes=filters)

# Offset
input_node_b = graph.nodes[-1]
param = Variable(init_bias(2))
graph.add_node(param)

node = Gate(Offset())
graph.add_node(node, parent_nodes=[input_node_b, param])

# Flatten
input_node_f = graph.nodes[-1]
flatten = Gate(Flatten())
graph.add_node(flatten, parent_nodes=[input_node_f])

# FC
input_node_fc = graph.nodes[-1]
Wc = init_weights(18, 5)
fc = Variable(Wc)
graph.add_node(fc)

matmul = Gate(MatMul())
graph.add_node(matmul, parent_nodes=[input_node_fc, fc])

input_node_c = graph.nodes[-1]

label_node = Output()
graph.add_node(label_node)

cost = Gate(Softmax())
graph.add_node(cost, parent_nodes=[input_node_c, label_node])

graph.forward(X, y)
print(graph.nodes[-1].state)
graph.backprop()

np.random.seed(1)
net = Sequential()

net.add_conv((4, 4, 3), 2)
net.add_flatten()
net.add_fc(18, 5, bias=False)
net.add_cost("softmax")

net.graph.forward(X, y)
print(net.graph.nodes[-1].state)
net.graph.backprop()
