import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, i_dim, kernel_dim, h_dim, nc, **kwargs):
        super(Discriminator, self).__init__()
        assert kernel_dim % 16 == 0, "kernel_dim has to be a multiple of 16"

        self.x = nn.Sequential(nn.Linear(h_dim*8*8+nc, 100),
                               nn.ReLU(inplace=True),
                               nn.Linear(100, 100),
                               nn.ReLU(inplace=True),
                               nn.Linear(100, 100),
                               nn.ReLU(inplace=True),
                               nn.Linear(100, 100),
                               nn.ReLU(inplace=True),
                               nn.Linear(100, 1))
        self.init()

    def init(self):
        for layer in self.x:
            if 'Linear' in layer.__class__.__name__:
                nn.init.kaiming_normal_(layer.weight, mode='FAN_IN', nonlinearity='relu')
                layer.bias.data.fill_(0)

    def forward(self, x, c):
        o = torch.cat((x.view(x.shape[0], -1), c.squeeze()), 1)
        return self.x(o).squeeze()


class GaussianLayer(nn.Module):
    def __init__(self):
        super(GaussianLayer, self).__init__()

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x)
        return x


class Classifier(nn.Module):
    def __init__(self, i_dim, h_dim, nc, **kwargs):
        super(Classifier, self).__init__()

        self.x = nn.Sequential(nn.InstanceNorm2d(3),
                               nn.Conv2d(i_dim, h_dim, 3, 1, 1),
                               nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True),
                               nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                               nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True),
                               nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                               nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True),
                               nn.MaxPool2d(2, 2),
                               nn.Dropout(0.5),
                               GaussianLayer(),
                               nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                               nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True),
                               nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                               nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True),
                               nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                               nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True),
                               nn.MaxPool2d(2, 2),
                               nn.Dropout(0.5),
                               GaussianLayer())

        self.mlp = nn.Sequential(nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                                 nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                                 nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                                 nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.AvgPool2d(8, 1),
                                 nn.Conv2d(h_dim, nc, 1, 1, 0),
                                 nn.BatchNorm2d(nc, momentum=0.99, eps=1e-3))
        self.init()

    def init(self):
        for layer in self.x:
            if 'Conv2d' in layer.__class__.__name__:
                nn.init.kaiming_normal_(layer.weight, mode='FAN_IN')
                layer.bias.data.fill_(0)

    def forward(self, x):
        o = self.x(x)
        return self.mlp(o).view(x.shape[0], -1)
