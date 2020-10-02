import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, h_dim, nc, **kwargs):
        super(Discriminator, self).__init__()

        self.x = nn.Sequential(nn.Linear(h_dim+nc, 100),
                               nn.ReLU(inplace=True),
                               nn.Linear(100, 1))
        self.init()

    def init(self):
        for layer in self.x:
            if 'Linear' in layer.__class__.__name__:
                nn.init.kaiming_normal_(layer.weight, mode='FAN_IN', nonlinearity='relu')
                layer.bias.data.fill_(0)

    def forward(self, x, c):
        o = torch.cat((x.view(x.shape[0], -1), c), 1)
        return self.x(o).squeeze()


class GaussianLayer(nn.Module):
    def __init__(self):
        super(GaussianLayer, self).__init__()

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x)
        return x


class Classifier(nn.Module):
    def __init__(self, h_dim, nc, z_dim, **kwargs):
        super(Classifier, self).__init__()

        self.x = nn.Sequential(nn.Linear(z_dim, h_dim),
                               nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True),
                               nn.Linear(h_dim, h_dim),
                               nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True),
                               nn.Dropout(0.5),
                               GaussianLayer(),
                               nn.Linear(h_dim, h_dim),
                               nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True),
                               nn.Linear(h_dim, h_dim),
                               nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True),
                               nn.Dropout(0.5),
                               GaussianLayer())

        self.mlp = nn.Sequential(nn.Linear(h_dim, h_dim),
                                 nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-3),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Linear(h_dim, h_dim),
                                 nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-3),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Linear(h_dim, h_dim),
                                 nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-3),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Linear(h_dim, nc),
                                 nn.LayerNorm(nc, elementwise_affine=False, eps=1e-3))

    def forward(self, x):
        o = self.x(x)
        return self.mlp(o)
