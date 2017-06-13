"""
Computational graph.
"""

import numpy as np


###############################################################################
# Computational graph


class ComputationalGraph(object):

    """Computational Graph.

    Class for building and calculating computational graphs. This
    implementation assumes all node objects are permeable.
    """

    def __init__(self):
        self.flag = 0
        self.n_nodes = 0
        self.nodes = list()
        self._input_nodes = list()
        self._output_nodes = list()

    def add_node(self,
                 node,
                 parent_nodes=None,
                 input_node=False,
                 label_node=False):
        """Add node to graph.

        Args
            node (Node): instance of the Node class
            parent_nodes (list, None): list of input nodes, or None
            input_node (bool): whether node expects the input data
            label_node (bool): whether node expect the label data
        """
        # Register if node requires external input
        if input_node:
            self._input_nodes.append(node)
        if label_node:
            self._output_nodes.append(node)

        # Set input and outputs
        if parent_nodes is not None:
            if not isinstance(parent_nodes, list):
                parent_nodes = [parent_nodes]

            for inp in parent_nodes:
                inp.set_output(node)
                node.set_input(inp)

        # Add node to graph
        self.nodes.append(node)

        # Increment counter
        self.n_nodes += 1

    def sorted_topologically(self, reverse=False):

        """Generates nodes according to forward dependencies.

        Args
            reverse (bool): whether to generate nodes in reverse.

        """
        T = self._sort_topologically()
        n = len(T) - 1

        for i in range(self.n_nodes):
            if reverse:
                i = n - i
            yield T[i]

    def _sort_topologically(self):
        """Return list of nodes sorted topologically for forward pass"""
        T = {0: self.nodes[0]}

        for i in range(self.n_nodes):
            n = T[i]
            outputs = n.get_output()

            # Current key counter
            s = max(T.keys())

            # Add all node specific inputs (parameters)
            inputs = [inp
                      for out in outputs
                      if out is not None
                      for inp in out.inputs
                      ]
            for inp in inputs:
                if inp not in T.values():
                    s += 1
                    assert s not in T.keys()
                    T[s] = inp

            # Add all output nodes
            for out in outputs:
                if out is not None and out not in T.values():
                    s += 1
                    assert s not in T.keys()
                    T[s] = out

        return T

    def forward(self, X, y=None, train=True):
        """Forward pass through graph (topologically).

        Args
            X (array): input array.
            X (array): label array.
            train (bool): whether the forward pass if for training.
        """
        self._initialize(X, y)

        for node in self.sorted_topologically():
            if isinstance(node, Variable):
                continue

            node.forward(train=train)

        # Set flag to 1 (forward pass completed)
        self.flag = 1

    def backprop(self):
        """Backward calculation.

        Runs backprop on a forward pass of the graph.
        """
        if not self.flag == 1:
            raise ValueError("Need a forward pass to do backprop.")

        for node in self.sorted_topologically(reverse=True):
            if node in self._input_nodes:
                continue

            node.backprop()

    def _initialize(self, X, y):
        """Initialize graph on X and y."""
        for n in self._input_nodes:
            n.state = X

        if y is not None:
            for n in self._output_nodes:
                n.state = y

    def clear(self):
        """Clear cache."""
        self.flag = 0

        for node in self.nodes:
            if not isinstance(node, Variable):
                node.clear()

        for n in self._input_nodes:
            n.state = None


###############################################################################
# Nodes of the graph


class Node(object):

    """Node meta class.
    """

    def __init__(self):
        self.grad = None
        self.state = None
        self.inputs = list()
        self.outputs = list()

    def set_input(self, inputs):
        """Set input node."""
        if not isinstance(inputs, list):
            inputs = [inputs]
        for i in inputs:
            self.inputs.append(i)

    def set_output(self, output):
        """Set output node."""
        if not isinstance(output, list):
            output = [output]
        for out in output:
            self.outputs.append(out)

    def get_output(self):
        """Set output node."""
        return self.outputs

    def get_input(self):
        """Get state of input nodes."""
        return [i.state for i in self.inputs]

    def get_grad(self):
        """Get gradient of output node."""
        if len(self.outputs) > 0:
            out = [o.grad[self]
                   for o in self.outputs
                   if o.grad[self] is not None]
            if len(out) == 0:
                return None
            if len(out) == 1:
                return out[0]
            else:
                return np.add(*out)


class Variable(Node):

    """Variable node.

    Variable nodes contains either data or parameters. We only do
    backprop updates on Variable nodes.

    Args
        X (array, int, float): parameter
    """

    def __init__(self, X=None):
        super(Variable, self).__init__()
        self.state = X
        self.__cls__ = self.__class__

    def backprop(self):
        """Backpropagate through the node."""
        grad = self.get_grad()
        self.grad = grad


class Gate(Node):

    """Node for performing an operation on two imputs.

    Args
        param (array, int): parameters for the node
        operation (obj): instance of the node operation class
    """

    def __init__(self, operation):
        super(Gate, self).__init__()
        self.operation = operation
        self.__cls__ = self.operation.__class__

    def forward(self, train=True):
        """Compute the state of the node in a forward pass."""
        inputs = self.get_input()

        if hasattr(self.operation, "train"):
            self.operation.train = train

        self.state = self.operation.forward(*inputs)

    def backprop(self):
        """Backpropagate through the node."""
        inputs = self.get_input()
        state = self.state
        grad = self.get_grad()

        grad = self.operation.backprop(*inputs, state, grad)
        self.grad = dict(zip(self.inputs, grad))

    def clear(self):
        """Clear state and gradient cache."""
        self.state = self.grad = None
        try:
            self.operation.clear()
        except AttributeError:
            pass
