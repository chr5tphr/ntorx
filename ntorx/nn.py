#!/usr/bin/env python3
import torch
from torch import nn
from contextlib import contextmanager
from collections import OrderedDict

from .func import Softplus, softplus_relu_diff

class Sequential(nn.Sequential):
    """
    """
    @contextmanager
    def ondev(self, device):
        try:
            self._rmtodev(device)
            yield self
        finally:
            self._rmtodev(torch.device('cpu'))

    def _rmtodev(self, dev):
        rmparams = self.parameters()
        self.to(dev)
        for param in rmparams:
            del param

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return __class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

class Parallel(nn.Module):
    pass

class Concat(nn.Module):
    pass

class Multiplexer(nn.Module):
    pass

class BatchView(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self._shape = shape

    def forward(self, x):
        ishape = x.shape
        return x.view(ishape[0], *self._shape)

class PaSU(nn.Module):
    """Parameterized Softplus Unit
    """
    def __init__(self, features, init=1e2, epsilon=1e-5, relu=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init = init
        self.beta = torch.nn.Parameter(torch.Tensor(features))
        self.epsilon = epsilon
        self.relu = relu
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.beta, self._init)

    def relu(self, mode=True):
        self.relu = mode

    def forward(self, x):
        #retval = torch.nn.functional.relu(x)
        #if not self.relu:
        #    bcast = torch.clamp(self.beta, self.epsilon)[(None, slice(None)) + (None,)*(len(x.shape)-2)]
        #    retval += softplus_relu_diff(x, bcast)
        #return retval
        if self.relu:
            return torch.nn.functional.relu(x)
        else:
            return Softplus.apply(x, self.beta)

class Linear(nn.Module):
    """Abstract Class for linear layers with weights and biases.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'weight'):
            self.weight = None
        if not hasattr(self, 'bias'):
            self.bias = None

    @contextmanager
    def with_params(self, weight, bias):
        try:
            old_weight = self.weight.data
            old_bias = self.bias.data
            self.weight.data = weight
            self.bias.data = bias
            yield self
        finally:
            self.weight.data = old_weight
            self.bias.data = old_bias

_linears = {
    'Dense' : nn.Linear,
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'Conv3d': nn.Conv3d,
    'BatchNorm1d': nn.BatchNorm1d,
    'BatchNorm2d': nn.BatchNorm2d,
    'BatchNorm3d': nn.BatchNorm3d,
}

_lintypes = {name: type(name, (Linear, base), {}) for name, base in _linears.items()}

def __getattr__(name):
    try:
        return _lintypes[name]
    except KeyError:
        pass

    raise AttributeError("module '{}' has no attribute '{}'".format(__name__, name))

def __dir__():
    return sorted(list(_linears.keys()))
