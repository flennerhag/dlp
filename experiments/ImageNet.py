"""
A simple test of a convolutional network trained on imagenet.
"""
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from deeplearn.funcs import probmax
from deeplearn.train import Trainer
from deeplearn.opts import Nesterov
from deeplearn.networks import Sequential
from deeplearn.viz import plot_train_scores

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])


trainset = CIFAR10(root='../pytorch/./data',
                   train=True,
                   download=False,
                   transform=transform)

testset = CIFAR10(root='../pytorch/./data',
                  train=False,
                  download=False,
                  transform=transform)

trainloader = DataLoader(trainset,
                         batch_size=4,
                         shuffle=True,
                         num_workers=3)


testloader = DataLoader(testset,
                        batch_size=4,
                        shuffle=False,
                        num_workers=3)

def build_net():
    """Build network with or without dropout."""
    net = Sequential()
    net.add_conv((3, 3, 3), 6)
    net.add_conv((2, 2, 6), 6, 2, 0, activation="relu")
    net.add_conv((4, 4, 6), 8)
    net.add_conv((4, 4, 8), 6, 1, 0, activation="relu")
    net.add_flatten()
    net.add_fc(864, 150, activation="relu")
    net.add_fc(150, 10, bias=False)
    net.add_cost("softmax")
    return net

def preprocess(blob):

    N = len(blob.train_labels)

    X = blob.train_data.astype(np.float32)
    for i in range(N):
        x = X[i, ...]

        m = x.max()
        n = x.min()
        v = (x - n) / (m - n)
        X[i, ...] = v

    return X, np.array(blob.train_labels)

def err(y, p):
    """Error of network"""
    y = y.ravel()
    p = probmax(p)
    return (p != y).sum() / y.shape[0]

if __name__ == "__main__":

    xtrain, ytrain = preprocess(trainset)

    net = build_net()
    opt = Nesterov(net, lr=1e-4, u=0.7)
    trainer = Trainer(net,
                      opt,
                      batch_size=100,
                      )

    trainer.train(xtrain, ytrain, 10000)
