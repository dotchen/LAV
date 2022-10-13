import torch
from torch import nn
from torch.nn import functional as F

class Normalize(nn.Module):
    """ ImageNet normalization """
    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std), requires_grad=False)

    def forward(self, x):
        return (x - self.mean[None,:,None,None]) / self.std[None,:,None,None]

class Interpolate(nn.Module):
    def __init__(self, scale_factor=2.):
        super().__init__()

        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)
