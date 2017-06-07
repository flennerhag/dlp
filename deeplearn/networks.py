"""
Simple feed forward network.
"""

from .init import init_bias, init_weights
from .graph import ComputationalGraph, Node
from .funcs import MatAdd, MatMul, ReLu, PReLu, Sigmoid, DropOut


ACTIVATIONS = {'relu': ReLu,
               'prelu': PReLu,
               'sigmoid': Sigmoid,
               'linear': None}


class Network(object):

    """Network meta class.
    """


class Sequential(Network):

    def __init__(self, features):
        self.features = features
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

        # Connection
        if dropout:
            input_node = self.graph.n_nodes - 1
            node = Node(None, DropOut(*dropout_args))
            self.graph.add_node(node, input_node)

        # Activation
        input_node = self.graph.n_nodes - 1
        op = ACTIVATIONS[activation]

        if act_arg is None:
            node = Node(None, op())
        else:
            node = Node(act_arg, op(act_arg))
        self.graph.add_node(node, input_node)

    def predict(self, X):
        """Predict X."""

        self.graph.forward(X, train=False)
        p = self.graph.nodes[-2].state
        self.graph.clear()
        return p
