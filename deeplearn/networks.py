"""
Simple feed forward network.
"""

from .conv import Convolution, Offset
from .init import init_bias, init_weights, init_filter
from .graph import ComputationalGraph, Node
from .cost_func import Norm, Softmax, BernSig
from .funcs import MatAdd, MatMul, ReLu, PReLu, Sigmoid, DropOut


ACTIVATIONS = {'relu': ReLu,
               'prelu': PReLu,
               'sigmoid': Sigmoid,
               'linear': None}

COST = {'norm': Norm,
        'softmax': Softmax,
        'bernsig': BernSig}


class Network(object):

    """Network meta class.
    """


class Sequential(Network):

    def __init__(self):
        self.graph = ComputationalGraph()

    def add_fc(self,
               fan_in,
               fan_out,
               bias=True,
               input_node=None,
               dropout=False,
               dropout_args=(0.5, None, False),
               activation="relu",
               act_arg=None):
        """Add fully connected layer to network."""
        if input_node is None:
            input_node = self.graph.n_nodes - 1

        # MatMul gate
        W = init_weights(fan_in, fan_out)
        node = Node(W, MatMul())
        self.graph.add_node(node, input_node)

        # MatAdd gate
        if bias:
            input_node = self.graph.n_nodes - 1
            b = init_bias(fan_out, 0)
            node = Node(b, MatAdd())
            self.graph.add_node(node, input_node)

        if dropout:
            self._connect(dropout_args)
        self._activate(activation, act_arg)

    def add_conv(self,
                 filter_size,
                 n_filters,
                 stride=1,
                 pad=1,
                 bias=True,
                 input_node=None,
                 dropout=False,
                 dropout_args=(0.5, None, False),
                 activation="relu",
                 act_arg=None):
        """Add a convolutional layer."""
        if input_node is None:
            input_node = self.graph.n_nodes - 1

        # Convolution
        W = init_filter(*filter_size, n_filters)

        node = Node(W, Convolution(n_filters, stride, pad))
        self.graph.add_node(node, input_node)

        # bias (one per filter)
        if bias:
            input_node = self.graph.n_nodes - 1
            b = init_bias(n_filters).ravel()
            node = Node(b, Offset())
            self.graph.add_node(node, input_node)

        if dropout:
            self._connect(dropout_args)

        self._activate(activation, act_arg)

    def add_cost(self, cost_type):
        """Add cost function."""
        input_node = self.graph.n_nodes - 1

        cost = COST[cost_type]
        node = Node(None, cost())
        self.graph.add_node(node, input_node)

    def predict(self, X):
        """Predict X."""

        self.graph.forward(X, train=False)
        p = self.graph.nodes[-2].state
        self.graph.clear()
        return p

    def _connect(self, dropout_args):
        """Add dropout connection."""
        input_node = self.graph.n_nodes - 1

        node = Node(None, DropOut(*dropout_args))
        self.graph.add_node(node, input_node)

    def _activate(self, activation, act_arg):
        """Add activation"""
        input_node = self.graph.n_nodes - 1
        op = ACTIVATIONS[activation]

        if act_arg is None:
            node = Node(None, op())
        else:
            node = Node(act_arg, op(act_arg))

        self.graph.add_node(node, input_node)
