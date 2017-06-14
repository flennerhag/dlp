"""
Simple feed forward network.
"""

from .conv import Convolution, Offset
from .init import init_bias, init_weights, init_filter
from .graph import ComputationalGraph, Input, Output, Variable, Gate
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
        self._activations_ = list()

        # Initialize input node
        input_node = Input()
        self._activations_.append(input_node)
        self.graph.add_node(input_node)

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

        # Add weight matrix parameter to graph
        W = init_weights(fan_in, fan_out)
        param = Variable(W)
        self.graph.add_node(param)

        # Add MatMul Gate to graph
        if input_node is None:
            input_node = self._activations_[-1]

        matmul = Gate(MatMul())
        self.graph.add_node(matmul, parent_nodes=[input_node, param])

        if bias:
            input_node = self.graph.nodes[-1]

            # Add bias parameter
            b = init_bias(fan_out, 0)
            param = Variable(b)
            self.graph.add_node(param)

            matadd = Gate(MatAdd())
            self.graph.add_node(matadd, parent_nodes=[input_node, param])

        if dropout:
            # Add dropout connection
            self._connect(dropout_args)

        # Add activation
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
            input_node = self.graph.nodes[-1]

        # Convolution
        W = init_filter(*filter_size, n_filters)
        param = Variable(W)
        self.graph.add_node(param)

        node = Gate(Convolution(n_filters, stride, pad))
        self.graph.add_node(node, parent_nodes=[input_node, param])

        # bias (one per filter)
        if bias:
            input_node = self.graph.nodes[-1]
            b = init_bias(n_filters).ravel()
            param = Variable(b)
            self.graph.add_node(param)

            node = Gate(Offset())
            self.graph.add_node(node, parent_nodes=[input_node, param])

        if dropout:
            self._connect(dropout_args)

        self._activate(activation, act_arg)

    def add_cost(self, cost_type):
        """Add cost function."""
        label_node = Output()
        self.graph.add_node(label_node)

        cost = COST[cost_type]()
        cost = Gate(cost)
        input_node = self._activations_[-1]

        self.graph.add_node(cost, parent_nodes=[input_node, label_node])

    def predict(self, X):
        """Predict X."""

        self.graph.forward(X, train=False)
        p = self.graph.nodes[-2].state
        self.graph.clear()
        return p

    def _connect(self, dropout_args):
        """Add dropout connection."""
        input_node = [self.graph.nodes[-1]]

        connection = Gate(DropOut(*dropout_args))
        self.graph.add_node(connection, parent_nodes=input_node)

    def _activate(self, activation, act_arg):
        """Add activation"""
        input_node = [self.graph.nodes[-1]]
        op = ACTIVATIONS[activation]

        if act_arg is None:
            activation = Gate(op())
            self.graph.add_node(activation, parent_nodes=input_node)
        else:
            # Add parameters as variable nodes
            for arg in act_arg.values():
                param = Variable(arg)
                input_node.append(param)
                self.graph.add_node(param)

            activation = Gate(op(**act_arg))
            self.graph.add_node(activation, parent_nodes=input_node)

        self._activations_.append(activation)
