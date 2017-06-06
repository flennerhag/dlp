"""
Computational graph.
"""


class ComputationalGraph(object):
    """Computational Graph.

    Class for building and calculating computational graphs. This
    implementation assumes all node objects are permutable.
    """

    def __init__(self):
        self.flag = 0
        self.n_nodes = 1
        self.nodes = []

        self.nodes.append(InitNode())

    def add_node(self, node, parent):
        """Add node to graph.

        Args
            node (Node): instance of the Node class
            parent (int): integer index to the input node
        """
        # Check valid input
        node_idx = self.n_nodes
        assert parent < self.n_nodes

        # Add node to graph
        self.nodes.append(node)

        # Add linkages
        node.set_input(self, parent)
        self.nodes[parent].set_output(self, node_idx)

        self.n_nodes += 1

    def forward(self, X, y=None):
        """Forward calculation though the graph with input matrix X.

        Args
            X (array): Original input array data.
        """
        # If y is passed, update labels
        if y is not None:
            self.nodes[-1].param = y

        # Initialize graph on X.
        self.nodes[0].forward(X)

        for node in self.nodes:
            if node.__class__.__name__ == 'InitNode':
                # No forward pass to be done on the InitNode - only holds X.
                continue

            node.forward()

        # Set flag to 1 (forward pass completed)
        self.flag = 1

    def backprop(self):
        """Backward calculation.

        Runs backprop on a forward pass of the graph.
        """
        # Check that a forward pass has been made.
        if not self.flag == 1:
            raise ValueError("Need a forward pass to do backprop.")

        # Run through the graph backwards.
        for i in range(1, self.n_nodes):
            node = self.nodes[-i]

            if node.__class__.__name__ == 'InitNode':
                # No backprop on the InitNode - only holds X.
                pass

            node.backprop()

    def clear(self):
        """Clear cache."""
        for node in self.nodes:
            node.clear()

        # Reset calculation flag.
        self.flag = 0


class InitNode(object):
    """Initial input data node.

    Container class that holds X as input for subsequent nodes.
    """
    def __init__(self):
        self.param = None
        self.state = None
        self.output = None
        self.__cls__ = self.__class__

    def forward(self, X):
        """Set X as state."""
        self.state = X

    @staticmethod
    def set_input(G, i):
        """Vacuous."""
        pass

    def set_output(self, G, j):
        """Set output node."""
        self.output = G.nodes[j]

    def clear(self):
        """Clear input cache."""
        self.state = None


class Node(object):
    """Node for computational graph.

    Class containing operations on nodes (variables).

    Args
        param (array, int): parameters for the node
        operation (obj): instance of the node operation class
    """

    def __init__(self, param, operation):
        self.state = None
        self.grad_input = None
        self.grad_param = None
        self.input = None
        self.output = None
        self.param = param
        self.operation = operation
        self.__cls__ = self.operation.__class__

    def forward(self):
        """Compute the state of the node in a forward pass."""
        X = self.get_state()
        self.state = self.operation.forward(X, self.param)

    def backprop(self):
        """Backpropagate through the node."""
        X = self.get_state()
        W = self.param
        H = self.state
        G = self.get_grad()

        self.grad_input, self.grad_param = self.operation.backprop(X, W, H, G)

    def set_input(self, G, i):
        """Set input node."""
        self.input = G.nodes[i]

    def set_output(self, G, j):
        """Set output node."""
        self.output = G.nodes[j]

    def get_state(self):
        """Get state of input node."""
        return self.input.state

    def get_grad(self):
        """Get gradient of output node."""
        if self.output is not None:
            return self.output.grad_input

    def clear(self):
        """Clear state and gradient cache."""
        self.state = self.grad_input = self.grad_param = None
