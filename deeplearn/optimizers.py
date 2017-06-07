"""
Optimization routines.
"""


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

    def update(self, graph, i):
        """Update nodes in graph."""
        for node in self.graph.nodes:
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

        self._initialize()

    def _initialize(self):
        """Set up velocity vectors."""
