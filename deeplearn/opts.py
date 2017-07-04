"""
Optimization routines.
"""

import numpy as np
from .graph import Variable
from .networks import Network


class Optimizer(object):

    """Optimizer meta class.
    """
    def __init__(self, graph):
        if issubclass(graph.__class__, Network):
            self.graph = graph.graph
        else:
            self.graph = graph


class GradientDescent(Optimizer):

    """Gradient Descent.
    """

    def __init__(self, graph, lr=1e-3, decay=0.):
        super(GradientDescent, self).__init__(graph)
        self.lr = lr
        self.decay = decay

    def update(self):
        """Update nodes in graph."""
        for node in self.graph.nodes:
            if isinstance(node, Variable):
                node.state -= self.lr * node.grad

        self.lr *= (1. - self.decay)


class Momentum(Optimizer):

    """Standard Momentum.
    """

    def __init__(self, graph, lr=1e-3, u=0.9, decay=0.):

        super(Momentum, self).__init__(graph)
        self.lr = lr
        self.u = u
        self.decay = decay

        self.W = self._initialize()

    def _initialize(self):
        """Set up velocity vectors."""
        W = dict()

        for node in self.graph.nodes:
            if isinstance(node, Variable):
                W[node] = 0.

        return W

    def update(self):
        """Update nodes in graph."""
        for node in self.graph.nodes:
            if not isinstance(node, Variable):
                continue

            self.W[node] = self.u * self.W[node] - self.lr * node.grad
            node.state += self.W[node]

        self.lr *= (1. - self.decay)


class Nesterov(Momentum):

    """Nesterov momentum.
    """

    def __init__(self, graph, lr=1e-3, u=0.9, decay=0.):
        super(Nesterov, self).__init__(graph, lr, u, decay)

    def update(self):
        """Update nodes in graph."""
        for node in self.graph.nodes:
            if not isinstance(node, Variable):
                continue

            W_ = self.W[node]

            self.W[node] = self.u * W_ - self.lr * node.grad
            node.state += (1. + self.u) * self.W[node] - self.u * W_

        self.lr *= (1. - self.decay)


class RMSProp(Optimizer):

    """Nesterov momentum.
    """

    def __init__(self, graph, lr=1e-3, u=0.9, decay=0., e=1e-7):
        super(RMSProp, self).__init__(graph)
        self.lr = lr
        self.u = u
        self.e = e
        self.decay = decay
        self.W = self._initialize()

    def _initialize(self):
        """Set up velocity vectors."""
        W = dict()
        for node in self.graph.nodes:
            if isinstance(node, Variable):
                W[node] = 0.

        return W

    def update(self):
        """Update nodes in graph."""
        for node in self.graph.nodes:
            if not isinstance(node, Variable) or node.grad is None:
                continue

            W = np.multiply(node.grad, node.grad)
            self.W[node] = self.u * self.W[node] + (1. - self.u) * W

            x = self.lr * node.grad / (np.sqrt(self.W[node]) + self.e)
            node.state -= x

        self.lr *= (1. - self.decay)


class Adam(Optimizer):

    """Nesterov momentum.
    """

    def __init__(self, graph, lr=1e-3, b1=0.9, b2=0.999, decay=0., e=1e-7):
        super(Adam, self).__init__(graph)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.decay = decay
        self.W1, self.W2 = self._initialize()
        self.i = 1.

    def _initialize(self):
        """Set up velocity vectors."""
        W1 = dict()
        W2 = dict()
        for node in self.graph.nodes:
            if isinstance(node, Variable):
                W1[node] = 0.
                W2[node] = 0.

        return W1, W2

    def update(self):
        """Update nodes in graph."""
        for node in self.graph.nodes:
            if not isinstance(node, Variable):
                continue

            # First moment
            m1 = self.b1 * self.W1[node] + (1. - self.b1) * node.grad
            m1 /= (1. - self.b1**self.i)

            # Second moment
            grad = np.multiply(node.grad, node.grad)
            m2 = self.b2 * self.W2[node] + (1. - self.b2) * grad
            m2 /= (1. - self.b2**self.i)

            node.state -= self.lr * m1 / (np.sqrt(m2) + self.e)

        self.lr *= (1. - self.decay)
        self.i += 1.
