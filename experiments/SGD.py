import numpy as np

from deeplearn.train import Trainer, GradientDescent, Momentum, Nesterov, \
    RMSProp, Adam
from deeplearn.networks import Sequential

from deeplearn.funcs import sigmoid
from deeplearn.viz import plot_train_scores

from sklearn.datasets import make_classification

###############################################################################
# Data and Network size

OBS = int(1e6)
ITERS = 400
STEPSIZE = 10

FEATS_0 = 10
FEATS_1 = 1000
FEATS_2 = 2000
FEATS_3 = 6
FEATS_4 = 500
FEATS_5 = 12

###############################################################################
# Data
X, y = make_classification(OBS, FEATS_0, 5)

#X -= X.mean(axis=0)
#X /= X.std(axis=0)

###############################################################################
# Network
def build_net():
    """Generate network"""
    net = Sequential()
    net.add_fc(FEATS_0, FEATS_1, normalize=True)
    net.add_fc(FEATS_1, FEATS_2, normalize=True)
    net.add_fc(FEATS_2, FEATS_3, normalize=True)
    net.add_fc(FEATS_3, FEATS_4, normalize=True)
    net.add_fc(FEATS_4, FEATS_5)
    net.add_fc(FEATS_5, 1)
    net.add_cost("norm")
    return net

###############################################################################
# Stochastic Gradient Descent


def acc(y, p, sig=False, C=0.5):
    """Accuracy of network"""
    y = y.ravel()
    p = p.ravel()
    if sig:
        p = sigmoid(p)

    p = 1 * (p > C)
    return (p == y).sum() / y.shape[0]

opt1 = GradientDescent
opt2 = Momentum
opt3 = Nesterov
opt4 = RMSProp
opt5 = Adam

for name, opt in zip(["SGD", "Momentum", "Nesterov", "RMSProp", "Adam"],
                     [opt1, opt2, opt3, opt4, opt5]):

    net = build_net()

    sgd = Trainer(net,
                  opt(net.graph, lr=5e-4, decay=1e-7),
                  batch_size=100,
                  eval_size=1000,
                  eval_ival=STEPSIZE,
                  eval_metric=acc,
                  )

    sgd.train(X, y, ITERS)

    plot_train_scores(sgd, title=name)
