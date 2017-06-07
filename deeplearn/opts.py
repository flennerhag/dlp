"""
Optimization routines.
"""

from copy import copy


class Optimizer(object):

    """Optimizer meta class.
    """


class GradientDescent(Optimizer):

    """Gradient Descent.
    """

    def __init__(self, graph, lr, decay=0.):
        self.graph = graph
        self.lr = lr
        self.decay = decay

    def update(self):
        """Update nodes in graph."""
        for node in self.graph.nodes:
            if node.param is None or node.grad_param is None:
                continue

            node.param -= self.lr * node.grad_param

        self.lr *= (1. - self.decay)


class Momentum(Optimizer):

    """Standard Momentum.
    """

    def __init__(self, graph, lr, u, decay=0.):
        self.graph = graph
        self.lr = lr
        self.u = u
        self.decay = decay

        self.W = self._initialize()

    def _initialize(self):
        """Set up velocity vectors."""
        W = dict()

        for node in self.graph.nodes:

            if node.param is None:
                continue

            try:
                W[node] = node.param.copy()
            except AttributeError:
                W[node] = copy(node.param)

        return W

    def update(self):
        """Update nodes in graph."""
        for node in self.graph.nodes:
            if node.param is None or node.grad_param is None:
                continue

            self.W[node] = self.u * self.W[node] + node.grad_param
            node.param -= self.lr * self.W[node]

        self.lr *= (1. - self.decay)


class Nesterov(Momentum):

    """Nesterov momentum.
    """

    def __init__(self, graph, lr, u, decay=0.):
        super(Nesterov, self).__init__(graph, lr, u, decay)
        self.rho = self.lr * self.u

    def update(self):
        """Update nodes in graph."""
        for node in self.graph.nodes:
            if node.param is None or node.grad_param is None:
                continue

            W_ = self.W[node]

            self.W[node] = self.rho * self.W[node] - self.lr * node.grad_param
            node.param += (1 + self.rho) * self.W[node] - self.rho * W_

        self.lr *= (1. - self.decay)
