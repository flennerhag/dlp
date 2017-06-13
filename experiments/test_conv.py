"""
Test run of a convolutional net
"""

import numpy as np
from deeplearn.train import Trainer
from deeplearn.opts import GradientDescent
from deeplearn.networks import Sequential


X = np.arange(50).reshape(5, 5, 2)
y = np.ones(1)

W = np.random.rand(6, 2, 2)

net = Sequential()

net.add_conv((3, 2, 2), 2, bias=True)
net.add_cost("norm")


net.graph.forward(X, y)
net.graph.backprop()

