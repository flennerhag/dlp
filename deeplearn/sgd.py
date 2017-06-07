"""
Stochastic gradient descent routine.
"""

import numpy as np
import matplotlib.pyplot as plt

from .cost_func import Cost


class Trainer(object):

    """Network trainer meta class.
    """


class SGD(Trainer):

    """Stochastic Gradient Descent trainer.
    """

    def __init__(self,
                 graph,
                 learning_rate=0.1,
                 decay=0.,
                 momentum=0.,
                 batch_size=200,
                 shuffle=True,
                 eval_size=None,
                 eval_ival=10,
                 eval_metric=None,
                 verbose=True):
        self.graph = graph
        self.learning_rate = learning_rate
        self.decay = decay
        self.batch_size = batch_size
        self.momentum = 0
        self.eval_size = eval_size
        self.eval_ival = eval_ival
        self.eval_metric = eval_metric
        self.shuffle = shuffle
        self.verbose = verbose

        self.W = None
        self.norms = None
        self.V_ = None
        self.v_ = None
        self.train_score = None
        self.test_score = None
        self.loss = None

    def train(self, X, y, n_iter):
        """Run Stochastic Gradient Descent.

        Args
            X (array): full training data
            y (array): training labels
            n_iter (int): number of batches to process

        Returns
            None: prints loss and gradient norm
        """
        # Losses and gradient norm at each epoch
        self.loss = []
        self.test_score = []
        self.train_score = []
        self.norms = [{'param': [],
                       'grad': [],
                       'type': n.__cls__,
                       'node': n}
                      for n in self.graph.nodes]

        if self.momentum != 0.:
            self.W = {node: node.param.copy() for node in self.graph.nodes}

        if self.shuffle:
            X, y = self._shuffle(X, y)

        if self.eval_size is not None:
            self.V_, self.v_ = X[:self.eval_size], y[:self.eval_size]
            X, y = X[self.eval_size:], y[self.eval_size:]

        batches_for_full_pass = X.shape[0] // self.batch_size

        self._print_start()

        start = 0
        for i in range(n_iter):

            j = i % batches_for_full_pass

            if j == 0:
                # Need to reset batch process
                start = 0
                if self.shuffle:
                    X, y = self._shuffle(X, y)

            # Grab batch. This is a view so no copying
            stop = (start + self.batch_size)
            X_, y_ = X[start:stop], y[start:stop]
            start = stop

            self._run_batch(X_, y_, i)

            if self.verbose:
                self._print_update(i)

        # Clean up
        self._clean()

    def _run_batch(self, X, y, i):
        """Parameter update on one batch.

        Args
            X (array): input batch
            y (array): batch labels

        Returns
            L (float): current loss
            g (float): output layer's gradient norm
        """
        # Run a forward pass and backpropagate errors.
        self.graph.forward(X, y)
        self.graph.backprop()

        # Store loss data
        L = self.graph.nodes[-1].state
        self.loss.append(L)

        # Run gradient updating of parameters
        self._update(i)

        # Evaluate train and test set
        self._eval(i)

        # Clear cache
        self.graph.clear()

    def _update(self, i):
        """Update parameter."""
        for j, node in enumerate(self.graph.nodes):

            if node.param is None and not issubclass(node.__cls__, Cost):
                continue

            if issubclass(node.__cls__, Cost):
                # Register the gradient wrt the input for the cost function
                grad = self._get_norm(node.grad_input)
                self.norms[j]['grad'].append(grad)
            else:
                # Register parameter and gradient norms
                grad = self._get_norm(node.grad_param)
                size = self._get_norm(node.param)
                self.norms[j]['grad'].append(grad)
                self.norms[j]['param'].append(size)

                # Update parameters
                if self.momentum == 0.:
                    node.param -= self.learning_rate * node.grad_param
                else:
                    u = self.momentum
                    self.W[node] = u * self.W[node] + node.grad_param
                    node.param -= self.learning_rate * self.W[node]

        # Update learning hyper-parameters
        self.learning_rate *= (1. - self.decay)

    @staticmethod
    def _get_norm(X):
        """Calculate norm."""
        try:
            size = np.linalg.norm(X.ravel())
        except AttributeError:
            if isinstance(X, (float, int)):
                size = X
            else:
                raise ValueError("Cannot calculate norm on input of type %s" %
                                 X.__class__.__name__)
        return size

    def _score_graph(self):
            """Score current state of the graph."""
            if self.eval_metric is None:
                L = self.graph.nodes[-1].state
            else:
                p = self.graph.nodes[-2].state
                y = self.graph.nodes[-1].param
                L = self.eval_metric(y, p)

            return L

    def _eval(self, i):
        """Evaluate graph on train and test set with current params."""
        if (i % self.eval_ival == 0) and (self.eval_size is not None):

            # Score train set
            l = self._score_graph()
            self.train_score.append(l)

            # Score test set
            self.graph.forward(self.V_, self.v_)
            l = self._score_graph()
            self.test_score.append(l)

    @staticmethod
    def _shuffle(X, y):
        """Shuffle training data."""
        idx = np.random.permutation(X.shape[0])
        X = X[idx]
        y = y[idx]
        return X, y

    def _print_update(self, i):
        """Print batch message."""
        msg = "[%i] %1.3f| "
        arg = [i + 1, self.loss[-1]]

        if self.eval_size is not None:
            msg += "%1.2f:%1.2f |"
            arg.append(self.train_score[-1]); arg.append(self.test_score[-1])

        N = len(self.norms)
        j = 1
        for norm_dict in reversed(self.norms):
            k = N - j
            j += 1

            cls = norm_dict["type"]
            if issubclass(cls, Cost):
                # Here we only want the size of the gradient wrt the input
                f = norm_dict["grad"][-1]

            elif cls.__name__ == "InitNode":
                # Container class for input data is not of interest
                continue

            else:
                n = norm_dict["param"][-1]
                d = norm_dict["grad"][-1]
                f = d / n if n != 0 else d

            msg += "%i: %.3f|"
            arg.append(k); arg.append(f)

        print(msg % tuple(arg))

    def _print_start(self):
        """Print column headings.
        """
        msg = "EPOCH  LOSS |"
        if self.eval_size is not None:
            msg += " TRN : TST |"

        msg += "NODE: GRADIENT NORM ->"
        print(msg)

    def _clean(self):
        """Create output arrays and clear temporary variables."""
        self.W = self.V_ = self.v_ = None

        self.loss = np.array(self.loss, dtype=np.float32)

        for norm_dict in self.norms:
            for entry in ["grad", "param"]:
                n = norm_dict[entry]
                n = np.array(n, dtype=np.float32)

        if self.eval_size is not None:
            self.test_score = np.array(self.test_score, dtype=np.float32)
            self.train_score = np.array(self.train_score, dtype=np.float32)


class DSGD(SGD):

    """Visual Stochastic Gradient Descent Trainer.
    """

    def __init__(self, graph, **kwargs):
        super(DSGD, self).__init__(graph, verbose=False, **kwargs)
        self.f = None
        self.ax = None
        self.pars = []

        for node in self.graph.nodes:
            if isinstance(node.param, np.ndarray):
                if len(node.param.shape) > 1:
                    self.pars.append(node)

    def _run_batch(self, X, y, i):
        """Visualization of batch processing."""
        # Run a forward pass and backpropagate errors.
        self.graph.forward(X, y)
        self.graph.backprop()

        # Store loss data
        L = self.graph.nodes[-1].state
        g = self.graph.nodes[-1].grad_input
        g = np.sqrt(g.T.dot(g))[0, 0]
        self.loss.append(L)
        self.norm.append(g)

        # Run gradient updating of parameters
        for node in self.graph.nodes:
            if node.param is not None and node.grad_param is not None:
                self._update(node, i)

        msg = "[%4i] %.2f | "
        arg = [i, self.loss[-1]]

        for j, node in enumerate(self.graph.nodes):
            try:
                m = node.grad_param
            except:
                continue

            if m is not None:
                if not isinstance(m, np.ndarray):
                    m *= m
                else:
                    m = np.sqrt(np.mean(np.multiply(m, m)))

                add = "(%i) %s: %.4f | "
                msg += add
                arg.append(j)
                arg.append(node.operation.__class__.__name__)
                arg.append(m)

        print(msg % tuple(arg))

        # Clear cache
        self._eval(i)
        self.graph.clear()

