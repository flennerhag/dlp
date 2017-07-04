"""
Introduction to networks with PyTorch
http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# Create a Tensor
x = torch.ones((3, 2))

# Modify tensor to desired init format
x[0, 1] *= -2; x[1, 1] *= -1; x[2, 1] *= -4; x[2, 0] *= -3; x[0, 0] *= 2

print(x)

# Wrap x into a Variable
x = Variable(x, requires_grad=True)

# Some other Variable
z = Variable(torch.rand((3, 2)))

# Create a function f: X * Z -> R
J = (x + z).abs().sum()

# Backpropagate
# If J is not a scalar (1d), the backprop needs a matrix as input
J.backward()

# Get gradients
print(x.grad)

###############################################################################
# Network


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Parameters

        # 1 input image, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 4)

        # affine operation: y = Wx + b
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        # If the size is a square you can specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # Flattening
        x = x.view(-1, self.num_flat_features(x))

        # First FC
        x = self.fc1(x)
        x = F.relu(x)

        # Second FC
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)


###############################################################################
# Forward run

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

# Dummy data
y = torch.zeros(2, 10); y[0, 1] = 1; y[1, 5] = 1
y = Variable(y)

X = Variable(torch.randn(2,   # Samples
                         1,   # Channel
                         32,  # Height
                         32)  # Width
             )

# Run inp through the net
net.train()
p = net(X)
print(p)

# Score the predictions
criterion = nn.MSELoss()

loss = criterion(p, y)
print(loss)

###############################################################################
# Backprop
net.zero_grad()
print('conv1.bias.grad before backprop')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backprop')
print(net.conv1.bias.grad)

###############################################################################
# Update

print('conv1.bias before update')
print(net.conv1.bias)

# create your optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

optimizer.step() # Does the update

print('conv1.bias after update')
print(net.conv1.bias)
